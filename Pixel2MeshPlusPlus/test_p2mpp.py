# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import tensorflow as tf
import tflearn
import numpy as np
import pprint
import pickle
import os

from modules.models_p2mpp import MeshNet
from modules.config import execute
from utils.dataloader import DataFetcher
from utils.tools import construct_feed_dict
from utils.visualize import plot_scatter


def xyz_to_obj(xyz_path, face_path='./data/face3.obj'):
    obj_path = xyz_path.replace('.xyz', '.obj')
    xyzf = np.loadtxt(xyz_path)
    v = np.full((xyzf.shape[0], 1), 'v')
    face = np.loadtxt(face_path, dtype='|S32')
    out = np.vstack((np.hstack((v, xyzf)), face))
    np.savetxt(obj_path, out, fmt='%s', delimiter=' ')
    return obj_path


def main(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)

    print('=> pre-processing')
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)

    num_blocks = 3
    num_supports = 2
    placeholders = {
        'features': tf.placeholder(tf.float32, shape=(None, 3), name='features'),
        'img_inp': tf.placeholder(tf.float32, shape=(cfg.num_input_images, 224, 224, 3), name='img_inp'),
        'labels': tf.placeholder(tf.float32, shape=(None, 6), name='labels'),
        'view_num': tf.placeholder(tf.int32, shape=(), name='view_num'),
        'support1': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'support2': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'support3': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'faces': [tf.placeholder(tf.int32, shape=(None, 4)) for _ in range(num_blocks)],
        'edges': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks)],
        'lape_idx': [tf.placeholder(tf.int32, shape=(None, 10)) for _ in range(num_blocks)],
        'pool_idx': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks - 1)],
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),
        'sample_coord': tf.placeholder(tf.float32, shape=(43, 3), name='sample_coord'),
        'cameras': tf.placeholder(tf.float32, shape=(cfg.num_input_images, 5), name='Cameras'),
        'faces_triangle': [tf.placeholder(tf.int32, shape=(None, 3)) for _ in range(num_blocks)],
        'sample_adj': [tf.placeholder(tf.float32, shape=(43, 43)) for _ in range(num_supports)],
    }
    placeholders["num_input_images"] = cfg.num_input_images

    step = cfg.test_epoch
    root_dir = os.path.join(cfg.save_path, cfg.name)
    model_dir = os.path.join(cfg.save_path, cfg.name, 'models')
    predict_dir = os.path.join(cfg.save_path, cfg.name, 'predict', str(step))
    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)
        print('==> make predict_dir {}'.format(predict_dir))

    print('=> build model')
    model = MeshNet(placeholders, logging=True, args=cfg)

    print('=> load data')
    data = DataFetcher(
        file_list=cfg.test_file_path,
        data_root=cfg.test_data_path,
        image_root=cfg.test_image_path,
        mesh_root=cfg.test_mesh_root,
        is_val=True,
        num_input_images=cfg.num_input_images,
        view_indices=cfg.view_indices
    )
    data.setDaemon(True)
    data.start()

    print('=> initialize session')
    sesscfg = tf.ConfigProto()
    sesscfg.gpu_options.allow_growth = True
    sesscfg.allow_soft_placement = True
    sess = tf.Session(config=sesscfg)
    sess.run(tf.global_variables_initializer())

    model.load(sess=sess, ckpt_path=model_dir, step=step)

    pkl = pickle.load(open('data/iccv_p2mpp.dat', 'rb'))
    pkl2 = None
    if args.prior != "default":
        print(f"Using custom prior from: {args.prior}")
        pkl2 = pickle.load(open(f"data/{args.prior}.dat", "rb"))
    
    if pkl2:
        print("Using prior from mean_shape_prior.dat")
        pkl["coord"] = pkl2["coord"]
    print("Available keys in iccv_p2mpp.dat:", pkl.keys())
    feed_dict = construct_feed_dict(pkl, placeholders)

    test_number = data.number
    tflearn.is_training(False, sess)
    print('=> start test stage 2')

    for iters in range(test_number):
        result = data.fetch()
        if result is None:
            print(f"[{iters}] Skipping invalid sample.")
            continue

        img_all_view, labels, poses, faces, data_id, mesh = result  # Match training order

        # Skip bad inputs
        if isinstance(mesh, str) or isinstance(img_all_view, str) or isinstance(labels, str) or isinstance(poses, str):
            print(f"[{data_id}] Skipping due to malformed inputs.")
            continue

        # Unpack npz labels if needed
        if isinstance(labels, np.lib.npyio.NpzFile):
            if 'points' in labels:
                labels = labels['points']
            elif 'xyz' in labels:
                labels = labels['xyz']
            else:
                raise ValueError(f"No 'points' or 'xyz' key in label npz file: {data_id}")

        feed_dict.update({
            placeholders['img_inp']: img_all_view,
            placeholders['features']: mesh,
            placeholders['labels']: labels,
            placeholders['cameras']: poses,
            placeholders['view_num']: cfg.num_input_images
        })

        out1l, out2l = sess.run([model.output1l, model.output2l], feed_dict=feed_dict)

        label_path = os.path.join(predict_dir, data_id.replace('.npz', '_ground.xyz'))
        pred_path = os.path.join(predict_dir, data_id.replace('.npz', '_predict.xyz'))

        np.savetxt(label_path, labels)
        np.savetxt(pred_path, out2l)
        
        try:
            obj_path = xyz_to_obj(pred_path)
            print(f"Saved OBJ: {obj_path}")
        except Exception as e:
            print(f"Could not convert to OBJ for {data_id}: {e}")

        plot_scatter(pt=out2l, data_name=data_id, plt_path=predict_dir)

        print('Iteration {}/{}, Data id {}'.format(iters + 1, test_number, data_id))

    data.shutdown()
    print('CNN-GCN Optimization Finished!')


if __name__ == '__main__':
    print('=> set config')
    args = execute()
    pprint.pprint(vars(args))
    main(args)
