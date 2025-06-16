import numpy as np
import pickle
import threading
import queue
import sys
import cv2
import os

np.random.seed(123)

class DataFetcher(threading.Thread):
    def __init__(self, file_list, data_root, image_root, is_val=False, mesh_root=None):
        super(DataFetcher, self).__init__()
        self.stopped = False
        self.queue = queue.Queue(64)
        self.data_root = data_root
        self.image_root = image_root
        self.is_val = is_val

        self.pkl_list = []
        with open(file_list, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.pkl_list.append(line)
        self.index = 0
        self.mesh_root = mesh_root
        self.number = len(self.pkl_list)
        np.random.shuffle(self.pkl_list)

    def work(self, idx):
        # Example: pkl_item = pollen_17787_Yellow_iris_Iris_pseudacorus_pollen_grain_00.dat
        pkl_item = self.pkl_list[idx]
        pkl_path = os.path.join(self.data_root, pkl_item)
        pkl = pickle.load(open(pkl_path, 'rb'), encoding='bytes')

        # Use only the label part
        print("DEBUG pkl[1] type:", type(pkl[1]), "shape:", np.shape(pkl[1]))
        # Fix: flatten and take first 6 values
        label = np.array(pkl[1]).flatten()[:6]
        label = label[np.newaxis, :]  # Add batch dimension, shape (1, 6)
        print("DEBUG label type:", type(label), "label shape:", np.shape(label))

        # Remove .dat extension and get the sample folder
        base_name = pkl_item.replace('.dat', '')
        # Remove the 'pollen_' prefix if present to get the folder name
        if base_name.startswith('pollen_'):
            folder_name = base_name[len('pollen_'):]
        else:
            folder_name = base_name

        # The rendering directory is: image_root/<folder_name>/rendering/
        img_path = os.path.join(self.image_root, folder_name, 'rendering')

        camera_meta_path = os.path.join(img_path, 'rendering_metadata.txt')
        camera_meta_data = np.loadtxt(camera_meta_path)
        # Ensure camera_meta_data is always (N, 5)
        if camera_meta_data.ndim == 1:
            camera_meta_data = camera_meta_data.reshape(-1, 5)

        if self.mesh_root is not None:
            mesh_path = os.path.join(self.mesh_root, folder_name + '_predict.xyz')
            mesh = np.loadtxt(mesh_path)
        else:
            mesh = None

        imgs = np.zeros((3, 224, 224, 3))
        poses = np.zeros((3, 5))
        for idx_img, view in enumerate([0, 6, 7]):
            img_file = os.path.join(img_path, f"{str(view).zfill(2)}.png")
            img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[-1] == 4:
                img[np.where(img[:, :, 3] == 0)] = 255
            img = cv2.resize(img, (224, 224))
            img_inp = img.astype('float32') / 255.0
            imgs[idx_img] = img_inp[:, :, :3]
            poses[idx_img] = camera_meta_data[view]

        # Debug prints
        print("DEBUG imgs shape:", imgs.shape, "dtype:", imgs.dtype)
        print("DEBUG poses shape:", poses.shape, "dtype:", poses.dtype)
        print("DEBUG label type:", type(label), "label shape:", np.shape(label))
        print("DEBUG pkl_item:", pkl_item)
        print("DEBUG mesh shape:", np.shape(mesh) if mesh is not None else None)

        return imgs, label, poses, pkl_item, mesh

    def run(self):
        while self.index < 9000000 and not self.stopped:
            self.queue.put(self.work(self.index % self.number))
            self.index += 1
            if self.index % self.number == 0:
                np.random.shuffle(self.pkl_list)

    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        while not self.queue.empty():
            self.queue.get()