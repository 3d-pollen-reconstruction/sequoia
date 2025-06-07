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
        try:
            # Example: pkl_item = pollen_17787_Yellow_iris_Iris_pseudacorus_pollen_grain_00.dat
            pkl_item = self.pkl_list[idx]
            pkl_path = os.path.join(self.data_root, pkl_item)
            print("pkl_path:", pkl_path)
            pkl = pickle.load(open(pkl_path, 'rb'), encoding='bytes')
            label = "pollen_" + pkl_item.replace('.npz', '').split('_')[1]  # Extract label from filename
            label = np.array([label], dtype='object')  # Convert label to numpy array
            print("Label:", label)
            # load image file

            # Remove .dat extension and get the sample folder
            base_name = pkl_item.replace('.dat', '')
            # Remove the 'pollen_' prefix if present to get the folder name
            if base_name.startswith('pollen_'):
                folder_name = base_name[len('pollen_'):]
            else:
                folder_name = base_name

            # The rendering directory is: image_root/<folder_name>/rendering/
            img_path = os.path.join(self.image_root, folder_name, 'rendering')
            print("Image path:", img_path)
            print("Folder name:", folder_name)
            print("Base name:", base_name)

            camera_meta_path = os.path.join(img_path, 'rendering_metadata.txt')
            if not os.path.exists(camera_meta_path):
                print(f"Metadata file not found: {camera_meta_path}")
                self.stopped = True
                return None
            camera_meta_data = np.loadtxt(camera_meta_path)
            # Ensure camera_meta_data is always (N, 5)
            if camera_meta_data.ndim == 1:
                camera_meta_data = camera_meta_data.reshape(-1, 5)
            
            print("GIGI")
            print(self.mesh_root)

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
            print("DEBUG pkl type:", type(pkl), "pkl length:", len(pkl))
            print("DEBUG mesh shape:", np.shape(mesh) if mesh is not None else None)

            verts = pkl['coord']
            faces = pkl['faces_triangle']
            return imgs, label, poses, pkl_item, mesh, verts, faces
        except Exception as e: 
            print("Exception in DataFetcher.work:", e)
            import traceback; traceback.print_exc()
            self.stopped = True
            return None

    def run(self):
        while self.index < 9000000 and not self.stopped:
            result = self.work(self.index % self.number)
            if result is not None:
                self.queue.put(result)
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