import os
import numpy as np
import cv2
import tensorflow as tf
import tflearn
import pickle

from modules.models_mvp2m import MeshNetMVP2M as MVP2MNet
from modules.models_p2mpp import MeshNet as P2MPPNet
from modules.config import execute
from utils.tools import construct_feed_dict

def load_and_preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = img[..., ::-1]  # BGR to RGB
    img = img.astype(np.float32) / 255.0
    return img[np.newaxis, ...]

def main(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    np.random.seed(123)
    tf.set_random_seed(123)

    base_dir = r"C:\Users\super\Documents\Github\sequoia\data\pollen_augmented\pollen_test"
    output_dir = "results/predicted_meshes"
    os.makedirs(output_dir, exist_ok=True)

    cfg.nviews = cfg.num_input_images
    num_blocks = 3
    num_supports = 2

    placeholders = {
        'features': tf.placeholder(tf.float32, shape=(None, 3), name='features'),
        'img_inp': tf.placeholder(tf.float32, shape=(cfg.nviews, 224, 224, 3), name='img_inp'),
        'labels': tf.placeholder(tf.float32, shape=(None, 6), name='labels'),
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
        'cameras': tf.placeholder(tf.float32, shape=(cfg.nviews, 3, 5), name='Cameras'),
        'faces_triangle': [tf.placeholder(tf.int32, shape=(None, 3)) for _ in range(num_blocks)],
        'sample_adj': [tf.placeholder(tf.float32, shape=(43, 43)) for _ in range(num_supports)],
    }
    placeholders['num_input_images'] = cfg.num_input_images  # required for GraphProjection layer

    print("=> building model")
    model1 = MVP2MNet(placeholders, logging=True, args=cfg)
    model2 = P2MPPNet(placeholders, logging=True, args=cfg)

    sesscfg = tf.ConfigProto()
    sesscfg.gpu_options.allow_growth = True
    sesscfg.allow_soft_placement = True
    sess = tf.Session(config=sesscfg)
    sess.run(tf.global_variables_initializer())

    model1.load(sess=sess, ckpt_path='results/coarse_mvp2m/models', step=200)
    model2.load(sess=sess, ckpt_path='results/refine_p2mpp/models', step=230)

    pkl = pickle.load(open('data/iccv_p2mpp.dat', 'rb'))
    base_feed_dict = construct_feed_dict(pkl, placeholders)

    tflearn.is_training(False, sess)

    for obj_idx, obj_name in enumerate(sorted(os.listdir(base_dir))):
        obj_path = os.path.join(base_dir, obj_name)
        rgb_dir = os.path.join(obj_path, "rgb_resized")
        pose_dir = os.path.join(obj_path, "pose")

        if not os.path.exists(rgb_dir) or not os.path.exists(pose_dir):
            continue

        image_names = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
        if len(image_names) < cfg.nviews:
            print(f"⚠️ Skipping {obj_name}: only {len(image_names)} views")
            continue

        print(f"➡️ Reconstructing {obj_name}...")

        img_paths = [os.path.join(rgb_dir, image_names[i]) for i in range(cfg.nviews)]
        pose_paths = [os.path.join(pose_dir, os.path.splitext(image_names[i])[0] + ".txt") for i in range(cfg.nviews)]

        cams = []
        for path in pose_paths:
            pose = np.loadtxt(path).reshape(4, 4)
            R = pose[:3, :3]
            t = pose[:3, 3:4]
            fx = 229.6875  # fixed fx used in your intrinsics
            f = np.array([[fx], [fx], [fx]])
            cams.append(np.concatenate([R, t, f], axis=1))  # shape (3, 5)

        cameras = np.stack(cams)
        images = np.concatenate([load_and_preprocess(p) for p in img_paths], axis=0)

        feed_dict = dict(base_feed_dict)
        feed_dict.update({
            placeholders['img_inp']: images,
            placeholders['labels']: np.zeros([10, 6]),
            placeholders['cameras']: cameras,
        })

        stage1_out = sess.run(model1.output3, feed_dict=feed_dict)
        feed_dict[placeholders['features']] = stage1_out
        verts = sess.run(model2.output2l, feed_dict=feed_dict)

        verts = np.hstack((np.full((verts.shape[0], 1), 'v'), verts))
        faces = np.loadtxt("data/face3.obj", dtype="|S32")
        mesh = np.vstack((verts, faces))

        out_path = os.path.join(output_dir, f"{obj_name}_v{cfg.nviews}.obj")
        np.savetxt(out_path, mesh, fmt="%s", delimiter=" ")
        print(f"✅ Saved to {out_path}")

    print("✅ All done.")

if __name__ == '__main__':
    print("=> set config")
    args = execute()
    main(args)
