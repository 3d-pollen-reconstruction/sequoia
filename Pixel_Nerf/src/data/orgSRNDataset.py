import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
from util import get_image_to_tensor_balanced, get_mask_to_tensor
import warnings # <--- HIER IMPORTIEREN
warnings.filterwarnings('ignore', category=UserWarning)


class SRNDataset(torch.utils.data.Dataset):
    """
    Dataset from SRN (V. Sitzmann et al. 2020)
    """

    def __init__(
        self, datadir, stage="train", image_size=(128, 128), world_scale=1.0
    ):
        """
        :param datadir: path to dataset root directory (e.g., .../pollen)
        :param stage: train | val | test
        :param image_size: result image size (resizes if different)
        :param world_scale: amount to scale entire world by
        """
        super().__init__()
        self.path = datadir  # WICHTIG: Speichere den Root-Pfad korrekt
        self.stage = stage
        self.image_size = image_size
        self.world_scale = world_scale

        # Ermittle den list_prefix (Kategorienamen)
        # Verwende den Basisnamen des datadir als Kategorie/Prefix
        self.list_prefix = os.path.basename(self.path)
        if not self.list_prefix: # Behandelt Fall, falls datadir mit Slash endet
             self.list_prefix = os.path.basename(os.path.dirname(self.path))
        print(f"INFO: Using list_prefix derived from datadir: '{self.list_prefix}'")

        self.dataset_name = self.list_prefix # Verwende den ermittelten Prefix als Namen
        self.base_path = os.path.join(self.path, self.list_prefix + "_" + self.stage)
        # ---------------------------------

        print("Loading SRN dataset", self.base_path, "name:", self.dataset_name)

        # --- Prüfen, ob der Basispfad existiert ---
        if not os.path.isdir(self.base_path):
             error_msg = f"ERROR: Base path for split does not exist or is not a directory: {self.base_path}"
             print(error_msg)
             raise FileNotFoundError(f"SRN dataset base path not found: {self.base_path}")
        # --------------------------------

        # Entferne oder kommentiere die is_chair Logik aus, falls nicht benötigt
        # is_chair = "chair" in self.dataset_name
        # if is_chair and stage == "train":
        #     tmp = os.path.join(self.base_path, "chairs_2.0_train")
        #     if os.path.exists(tmp):
        #         self.base_path = tmp

        # --- Lade Liste der Objekt-IDs ---
        self.list_path = os.path.join(self.path, self.list_prefix + "_" + self.stage + ".lst")
        if not os.path.exists(self.list_path):
             print(f"INFO: Split file {self.list_path} does not exist, using objects found in base path {self.base_path}")
             try:
                self.ids = sorted([d for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, d))])
             except Exception as e:
                 print(f"ERROR: Could not list directories in {self.base_path}: {e}")
                 self.ids = []
        else:
             print(f"INFO: Loading object list from {self.list_path}")
             with open(self.list_path, "r") as f:
                 self.ids = sorted([x.strip() for x in f.readlines()])
        # -----------------------------

        # --- Prüfe Anzahl der Objekte ---
        self.n_objs = len(self.ids)
        print(f"Found {self.n_objs} object IDs for stage '{self.stage}'.")
        if self.n_objs == 0:
             error_msg = f"ERROR: No objects found for stage '{self.stage}' in {self.base_path} using list_prefix '{self.list_prefix}'"
             print(error_msg)
             raise ValueError(f"No objects found for SRN dataset stage {self.stage}")
        # ----------------------------

        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()

        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )

        # --- Setze Near/Far Ebenen (Verwende SRN-Standardwerte oder passe manuell an) ---
        # Diese Werte könnten für Ihre spezifischen Daten/Skalierung angepasst werden müssen
        self.z_near = 0.8
        self.z_far = 5.0
        self.lindisp = True # SRN verwendet normalerweise lineares Tiefen-Sampling
        # --------------------------------------------------------------------------
        # Falls near_far.txt existiert, verwende diese Werte
        sample_obj = self.ids[0]
        nf_path = os.path.join(self.base_path, sample_obj, "near_far.txt")
        if os.path.exists(nf_path):
            with open(nf_path, "r") as f:
                self.z_near, self.z_far = [float(x) for x in f.readline().split()]
            print(f"INFO: Loaded near/far from {nf_path}: z_near = {self.z_near}, z_far = {self.z_far}")
        else:
            print(f"WARNUNG: near_far.txt nicht gefunden bei {nf_path}, benutze Defaultwerte.")
        print(f"Using z_near: {self.z_near}, z_far: {self.z_far}, lindisp: {self.lindisp}")

# --- ENDE DES CODES ZUM ERSETZEN ---

    def __len__(self):
        return self.n_objs

    # --- START CODE ZUM ERSETZEN IN src/data/SRNDataset.py ---

    # Ersetzen Sie die GESAMTE __getitem__ Methode durch diese:
    def __getitem__(self, index):
        # Hole Objekt-ID für diesen Index
        object_id = self.ids[index]
        # Konstruiere Pfad zu den Daten dieses Objekts im aktuellen Split
        dir_path = os.path.join(self.base_path, object_id) # z.B., .../pollen/pollen_train/17767...

        # --- Korrekte Ermittlung des Intrinsics-Pfades ---
        intrin_path = os.path.join(dir_path, "intrinsics.txt") # Intrinsics-Pfad pro Objekt
        if not os.path.exists(intrin_path):
            # Gib einen aussagekräftigen Fehler aus, wenn die Datei fehlt
            raise FileNotFoundError(f"Intrinsics file not found for object {object_id} at: {intrin_path}")
        # ----------------------------------------------------

        # Suche nach Bildern und Posen für dieses Objekt
        rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*.png"))) # Verwende .png
        pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*.txt"))) # Verwende .txt

        if not rgb_paths or not pose_paths:
            # Fehler, wenn keine Bilder oder Posen gefunden wurden
            raise FileNotFoundError(f"No images (.png) or poses (.txt) found in {dir_path}/rgb or {dir_path}/pose for object {object_id}")

        # Stelle sicher, dass die Anzahl übereinstimmt, nimm das Minimum
        num_views = min(len(rgb_paths), len(pose_paths))
        if len(rgb_paths) != len(pose_paths):
            print(f"WARNUNG: Unterschiedliche Anzahl von Bildern ({len(rgb_paths)}) und Posen ({len(pose_paths)}) für Objekt {object_id}. Verwende Minimum ({num_views}).")
            # Kürze die längere Liste, um sie anzugleichen
            rgb_paths = rgb_paths[:num_views]
            pose_paths = pose_paths[:num_views]

        # Lade Intrinsics aus der Datei
        with open(intrin_path, "r") as intrinfile:
            lines = intrinfile.readlines()
            try:
                # Liest fx, cx, cy aus Zeile 1 (angepasst an Ihr Format)
                focal, cx, cy, _ = map(float, lines[0].split())
                # Liest width, height aus der letzten Zeile (angepasst an Ihr Format)
                # Annahme: Letzte Zeile ist "width height"
                height, width = map(int, lines[-1].strip().split())
            except Exception as e:
                # Fehler beim Parsen der Intrinsics-Datei
                raise ValueError(f"Error parsing intrinsics file {intrin_path}: {e}")

        # Lade alle Bilder, Posen, Masken für die Views dieses Objekts
        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        # Iteriere über die gefundene Anzahl von Views
        for i in range(num_views):
            rgb_path = rgb_paths[i]
            pose_path = pose_paths[i]

            # Lade Bild
            # Lade Bild mit Alphakanal und setze weißen Hintergrund
            try:
                img_rgba = imageio.imread(rgb_path)
                if img_rgba.shape[-1] == 4:
                    alpha = img_rgba[..., 3:] / 255.0
                    img = img_rgba[..., :3] * alpha + 255 * (1 - alpha)  # weißer Hintergrund
                    img = img.astype(np.uint8)
                else:
                    img = img_rgba[..., :3]
            except Exception as e:
                print(f"WARNUNG: Überspringe Ansicht {i} für Objekt {object_id} wegen Fehler beim Laden des Bildes: {e}")
                continue


            # Konvertiere Bild und erstelle Maske
            img_tensor = self.image_to_tensor(img)
            # Maskengenerierung nimmt an, dass Weiss (255) transparent ist
            #mask = (img != 255).all(axis=-1)[..., None].astype(np.uint8) * 255
            dist = np.linalg.norm(img.astype(np.int16) - 255, axis=-1)
            mask = (dist > 5)[..., None].astype(np.uint8) * 255
            mask_tensor = self.mask_to_tensor(mask)

            if mask_tensor.sum() < 100:
                print(f"⚠️  Maske fast leer für {rgb_path}")

            # Lade Pose
            try:
                # Lade flache Pose und forme sie um
                pose_flat = np.loadtxt(pose_path, dtype=np.float32)
                if pose_flat.size != 16:
                    raise ValueError(f"Pose file {pose_path} does not contain 16 values.")
                pose = torch.from_numpy(pose_flat.reshape(4, 4))
            except Exception as e:
                print(f"WARNUNG: Überspringe Ansicht {i} für Objekt {object_id} wegen Fehler beim Laden der Pose: {e}")
                continue # Überspringe diese Ansicht, wenn die Pose nicht geladen werden kann

            # Wende Koordinatentransformation an
            pose = pose @ self._coord_trans

            # Berechne Bounding Box aus der Maske
            rows = np.any(mask[...,0], axis=1) # mask hat shape H,W,1 -> H,W nötig
            cols = np.any(mask[...,0], axis=0)
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]
            if len(rnz) == 0 or len(cnz) == 0: # Prüfe, ob Maske nicht leer ist
                print(f"WARNUNG: Überspringe Ansicht {i} für Objekt {object_id} wegen leerer Maske/BBox.")
                # Alternative: Standard-BBox statt Überspringen
                # bbox = torch.tensor([0, 0, width-1, height-1], dtype=torch.float32)
                continue # Überspringe, wenn BBox nicht berechnet werden kann
            rmin, rmax = rnz[[0, -1]]
            cmin, cmax = cnz[[0, -1]]
            bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)
            # ---- Ende BBox Berechnung ----

            # Füge gültige Daten zu den Listen hinzu
            all_imgs.append(img_tensor)
            all_masks.append(mask_tensor)
            all_poses.append(pose)
            all_bboxes.append(bbox)

        # Prüfe, ob überhaupt gültige Views für dieses Objekt geladen wurden
        if not all_imgs:
            # Dieser Fehler sollte nicht auftreten, wenn die Ordnerstruktur korrekt ist
            # und zumindest eine Ansicht pro Objekt geladen werden kann.
            raise RuntimeError(f"No valid views could be loaded for object {object_id} at {dir_path}. Check image/pose files.")

        # Stacke die Tensoren für alle Views dieses Objekts
        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_masks = torch.stack(all_masks)
        all_bboxes = torch.stack(all_bboxes)

        # Skaliere Bilder/Masken/BBoxen auf die Zielgrösse, falls nötig
        if tuple(all_imgs.shape[-2:]) != self.image_size:
            print(f"INFO: Resizing images for object {object_id} from {tuple(all_imgs.shape[-2:])} to {self.image_size}")
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            cx *= scale
            cy *= scale
            all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        # Wende Weltskalierung an, falls gesetzt
        if self.world_scale != 1.0:
            focal *= self.world_scale
            all_poses[:, :3, 3] *= self.world_scale

        # Stelle sicher, dass focal und c skalare bzw. 2-elementige Tensoren sind
        focal = torch.tensor(focal, dtype=torch.float32).squeeze()
        center_point = torch.tensor([cx, cy], dtype=torch.float32).squeeze()

        # Erstelle das Ergebnis-Dictionary für dieses Objekt
        result = {
            "path": dir_path, # Pfad zum Objektordner
            "img_id": index, # Index des Objekts in der Liste dieses Splits
            "focal": focal,  # Brennweite als Tensor
            "c": center_point, # Bildzentrum (cx, cy) als Tensor
            "images": all_imgs, # (NV, 3, H, W)
            "masks": all_masks,  # (NV, 1, H, W) - wichtig für einige Verluste/Visualisierungen
            "bbox": all_bboxes,  # (NV, 4) - wichtig für BBox-Sampling
            "poses": all_poses,  # (NV, 4, 4)
        }
        return result

# --- ENDE DES CODES ZUM ERSETZEN ---