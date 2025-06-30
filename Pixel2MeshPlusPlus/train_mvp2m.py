# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import tensorflow as tf
import tflearn
import numpy as np
import pprint
import pickle
import shutil
import os

from modules.models_mvp2m import MeshNetMVP2M
from utils.dataloader import DataFetcher
from utils.tools import construct_feed_dict
from modules.config import execute
from utils.visualize import plot_scatter


def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    # ---------------------------------------------------------------
    # Set random seed
    print("=> pre-porcessing")
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)
    # ---------------------------------------------------------------
    num_blocks = 3
    num_supports = 2
    placeholders = {
        "features": tf.placeholder(tf.float32, shape=(None, 3), name="features"),
        #'img_inp': tf.placeholder(tf.float32, shape=(3, 224, 224, 3), name='img_inp'),
        "img_inp": tf.placeholder(
            tf.float32, shape=(cfg.num_input_images, 224, 224, 3), name="img_inp"
        ),
        "labels": tf.placeholder(tf.float32, shape=(None, 6), name="labels"),
        "view_num": tf.placeholder(tf.int32, shape=(), name="view_num"),
        "support1": [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        "support2": [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        "support3": [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        "faces": [tf.placeholder(tf.int32, shape=(None, 4)) for _ in range(num_blocks)],
        "edges": [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks)],
        "lape_idx": [
            tf.placeholder(tf.int32, shape=(None, 10)) for _ in range(num_blocks)
        ],  # for laplace term
        "pool_idx": [
            tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks - 1)
        ],  # for unpooling
        "dropout": tf.placeholder_with_default(0.0, shape=()),
        "num_features_nonzero": tf.placeholder(tf.int32),
        "sample_coord": tf.placeholder(tf.float32, shape=(43, 3), name="sample_coord"),
        "cameras": tf.placeholder(
            tf.float32, shape=(cfg.num_input_images, 5), name="Cameras"
        ),
        "faces_triangle": [
            tf.placeholder(tf.int32, shape=(None, 3)) for _ in range(num_blocks)
        ],
        "sample_adj": [
            tf.placeholder(tf.float32, shape=(43, 43)) for _ in range(num_supports)
        ],
    }
    placeholders["num_input_images"] = cfg.num_input_images


    root_dir = os.path.join(cfg.save_path, cfg.name)
    model_dir = os.path.join(cfg.save_path, cfg.name, "models")
    log_dir = os.path.join(cfg.save_path, cfg.name, "logs")
    plt_dir = os.path.join(cfg.save_path, cfg.name, "plt")
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
        print("==> make root dir {}".format(root_dir))
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        print("==> make model dir {}".format(model_dir))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        print("==> make log dir {}".format(log_dir))
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)
        print("==> make plt dir {}".format(plt_dir))
    summaries_dir = os.path.join(cfg.save_path, cfg.name, "summaries")
    train_loss = open("{}/train_loss_record.txt".format(log_dir), "a")
    train_loss.write("Net {} | Start training | lr =  {}\n".format(cfg.name, cfg.lr))
    # -------------------------------------------------------------------
    print("=> build model")
    # Define model
    model = MeshNetMVP2M(placeholders, logging=True, args=cfg)
    
    # Freeze encoder variables if requested
    if cfg.freeze_encoder:
        print("=> Freezing encoder variables")
        encoder_vars = [var for var in tf.trainable_variables() 
                       if any(keyword in var.name.lower() for keyword in ['encoder', 'resnet', 'vgg', 'conv2d', 'cnn'])]
        print(f"Found {len(encoder_vars)} encoder variables to freeze:")
        for var in encoder_vars[:5]:  # Show first 5 variables
            print(f"  {var.name}: {var.shape}")
        if len(encoder_vars) > 5:
            print(f"  ... and {len(encoder_vars) - 5} more")
        
        # Get non-encoder variables for training
        trainable_vars = [var for var in tf.trainable_variables() if var not in encoder_vars]
        print(f"Training {len(trainable_vars)} non-encoder variables")
        
        # Update the optimizer to only train non-encoder variables
        model.opt_op = model.optimizer.minimize(model.loss, var_list=trainable_vars)
    
    # ---------------------------------------------------------------
    print("=> load data")
    print("Train data file path:", cfg.train_file_path)
    print("Train data root path:", cfg.train_data_path)
    print("Train image root path:", cfg.train_image_path)
    print("Number of input images:", cfg.num_input_images)
    data = DataFetcher(
        file_list=cfg.train_file_path,
        data_root=cfg.train_data_path,
        image_root=cfg.train_image_path,
        is_val=False,
        num_input_images=cfg.num_input_images,
        view_indices=cfg.view_indices,
    )
    data.setDaemon(True)
    data.start()
    # ---------------------------------------------------------------
    print("=> initialize session")
    sesscfg = tf.ConfigProto()
    sesscfg.gpu_options.allow_growth = True
    sesscfg.allow_soft_placement = True
    sess = tf.Session(config=sesscfg)
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(
        summaries_dir, sess.graph, filename_suffix="train"
    )
    # ---------------------------------------------------------------
    if cfg.restore:
        print("=> load model")
        model.load(sess=sess, ckpt_path=model_dir, step=cfg.init_epoch)
    # ---------------------------------------------------------------
    # Load init ellipsoid and info about vertices and edges
    pkl = pickle.load(open("data/iccv_p2mpp.dat", "rb"))
    pkl2 = None
    if args.prior != "default":
        print(f"Using custom prior from: {args.prior}")
        pkl2 = pickle.load(open(f"data/{args.prior}.dat", "rb"))
    
    if pkl2:
        print("Using prior from mean_shape_prior.dat")
        pkl["coord"] = pkl2["coord"]
    print("Available keys in iccv_p2mpp.dat:", pkl.keys())

        
    # Construct Feed dict
    feed_dict = construct_feed_dict(pkl, placeholders)
    # ---------------------------------------------------------------
    train_number = data.number
    step = 0
    tflearn.is_training(True, sess)
    print("=> start train stage 1")
    for epoch in range(cfg.epochs):
        current_epoch = epoch + 1 + cfg.init_epoch
        epoch_plt_dir = os.path.join(plt_dir, str(current_epoch))
        if not os.path.exists(epoch_plt_dir):
            os.mkdir(epoch_plt_dir)
        mean_loss = 0
        all_loss = np.zeros(train_number, dtype="float32")
        for iters in range(train_number):
            step += 1
            # Fetch training data
            # need [img, label, pose(camera meta data), dataID]
            img_all_view, labels, poses, faces, data_id, mesh = data.fetch()
            if isinstance(labels, np.lib.npyio.NpzFile):
                if "points" in labels:
                    labels = labels["points"]
                elif "xyz" in labels:
                    labels = labels["xyz"]
                else:
                    raise ValueError(
                        f"No 'points' or 'xyz' key in label npz file: {data_id}"
                    )

            feed_dict.update({placeholders["img_inp"]: img_all_view})
            feed_dict.update({placeholders["view_num"]: cfg.num_input_images})

            feed_dict.update(
                {placeholders["labels"]: labels}  # Already normalized in .npz
            )

            feed_dict.update({placeholders["cameras"]: poses})
            feed_dict.update(
                {placeholders["faces"][0]: faces}
            )  # or as needed for your model
            print("labels shape:", labels.shape)
            print("Sample label points:\n", labels[:5])
            # ---------------------------------------------------------------
            _, dists, summaries, out1, out3, out3 = sess.run(
                [
                    model.opt_op,
                    model.loss,
                    model.merged_summary_op,
                    model.output1,
                    model.output2,
                    model.output3,
                ],
                feed_dict=feed_dict,
            )
            print("Expected label shape (should match vertex count):", labels.shape)
            print("Expected prediction shape (out3):", out3.shape)

            initial_verts = pkl[
                "coord"
            ]  # Shape should be (156, 3) for the base ellipsoid

            # ---------------------------------------------------------------
            all_loss[iters] = dists
            mean_loss = np.mean(all_loss[np.where(all_loss)])
            print(
                "Epoch {}, Iteration {}, Mean loss = {}, iter loss = {}, {}, data id {}".format(
                    current_epoch,
                    iters + 1,
                    mean_loss,
                    dists,
                    data.queue.qsize(),
                    data_id,
                )
            )
            train_writer.add_summary(summaries, step)
            # if cfg.train_data_path contains augmented as path every 5000 iterations plot
            
            if "augmented" in cfg.train_data_path:
                
                if (iters + 1) % 5000 == 0:
                    # === DEBUG OUTPUT ===
                    print("Shape of out3:", out3.shape)
                    print("X range:", np.min(out3[:, 0]), np.max(out3[:, 0]))
                    print("Y range:", np.min(out3[:, 1]), np.max(out3[:, 1]))
                    print("Z range:", np.min(out3[:, 2]), np.max(out3[:, 2]))
                    print("NaNs in out3:", np.isnan(out3).any())
                    print("All-zero vertices in out3:", np.all(out3 == 0, axis=1).sum())

                    # Plot model prediction
                    plot_scatter(
                        pt=out3, data_name="pred_" + data_id, plt_path=epoch_plt_dir
                    )
                    plot_scatter(
                        initial_verts,
                        data_name="init_ellipsoid.png",
                        plt_path=epoch_plt_dir,
                    )

                    # Plot ground truth label
                    if labels.shape[1] >= 3:
                        plot_scatter(
                            pt=labels[:, :3],
                            data_name="_label" + data_id,
                            plt_path=epoch_plt_dir,
                        )
                    else:
                        print("Warning: Labels do not contain XYZ coordinates.")
                        
            # else plot every 140 iterations
            elif (iters + 1) % 140 == 0:
                # === DEBUG OUTPUT ===
                print("Shape of out3:", out3.shape)
                print("X range:", np.min(out3[:, 0]), np.max(out3[:, 0]))
                print("Y range:", np.min(out3[:, 1]), np.max(out3[:, 1]))
                print("Z range:", np.min(out3[:, 2]), np.max(out3[:, 2]))
                print("NaNs in out3:", np.isnan(out3).any())
                print("All-zero vertices in out3:", np.all(out3 == 0, axis=1).sum())

                # Plot model prediction
                plot_scatter(
                    pt=out3, data_name="pred_" + data_id, plt_path=epoch_plt_dir
                )
                plot_scatter(
                    initial_verts,
                    data_name="init_ellipsoid.png",
                    plt_path=epoch_plt_dir,
                )


                # Plot ground truth label
                if labels.shape[1] >= 3:
                    plot_scatter(
                        pt=labels[:, :3],
                        data_name="_label" + data_id,
                        plt_path=epoch_plt_dir,
                    )
                else:
                    print("Warning: Labels do not contain XYZ coordinates.")

        # ---------------------------------------------------------------
        # Save model
        model.save(sess=sess, ckpt_path=model_dir, step=current_epoch)
        train_loss.write("Epoch {}, loss {}\n".format(current_epoch, mean_loss))
        train_loss.flush()
    # ---------------------------------------------------------------
    data.shutdown()
    print("CNN-GCN Optimization Finished!")


if __name__ == "__main__":
    print("=> set config")
    args = execute()
    pprint.pprint(vars(args))
    main(args)
