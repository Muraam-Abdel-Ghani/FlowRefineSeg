import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class EndoVisDataset(Dataset):
    def __init__(self, root_dir, sequences, id2color, has_greyscale_labels, instrument_ids, image_size=(512, 512)):
        """
        Args:
            root_dir (str): Path to the dataset root directory.
            extractor (callable): Feature extractor to preprocess images and masks.
            sequences (list): List of sequences to include.
            id2color_instruments (dict): Mapping of instrument class IDs to their respective RGB colors.
            instrument_ids (list): List of instrument class IDs.
            image_size (tuple): Tuple specifying the image size (H, W).
        """
        self.root_dir = root_dir
        self.sequences = sequences
        self.id2color = id2color
        self.has_greyscale_labels = has_greyscale_labels
        self.instrument_ids = instrument_ids
        self.image_size = image_size

        self.img_files = []
        self.ann_files = []

        self.sequence_labels = []  # NEW for temporal refinery

        # Collect image and annotation files from all sequences
        seq_id=0
        for seq in self.sequences:
            for frame_type in ["left_frames"]:
                img_dir = os.path.join(self.root_dir, seq, frame_type)
                ann_dir = os.path.join(self.root_dir, seq, "labels")

                img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
                
                # For kvasir, ends with .jpg
                #img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
                ann_files = sorted([f for f in os.listdir(ann_dir) if f.endswith(".png")])

                self.img_files.extend([os.path.join(img_dir, img_file) for img_file in img_files])
                self.ann_files.extend([os.path.join(ann_dir, ann_file) for ann_file in ann_files])
                
                seq_id+1
                for i in range(len(img_files)):
                    self.sequence_labels.append(seq_id)  # track which sequence
                

        # Transformations (resizing)
        self.resize_transform = transforms.Resize(self.image_size)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.img_files[idx]).convert("RGB")

        #comment this if computing frames per second ----
        if self.has_greyscale_labels:  
            segmentation_map = np.array(Image.open(self.ann_files[idx]))
        else:
            segmentation_map = np.array(Image.open(self.ann_files[idx]).convert("RGB"))
        # ---------

        # Resize image and mask to fixed size
        image = self.resize_transform(image)


        # comment if computing frames per second ------------------------------------------
        segmentation_map_2d = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1]), dtype=np.uint8) #shape[0] = height, shape[1] = width in numpy array


        for id, color in self.id2color.items():
            # for parts (colored maps)
            if self.has_greyscale_labels:               
                # for greyscale maps (with only 2 dimensions)
                # segmentation_map_2d[np.array(segmentation_map) == color] = id
                segmentation_map_2d[segmentation_map == color] = id
            else: 
                # for RGB maps (with 3 dimensions, third contains RGB)
                # segmentation_map_2d[(np.array(segmentation_map) == color).all(axis=2)] = id
                segmentation_map_2d[(segmentation_map == color).all(axis=2)] = id

        # Clamp the labels to ensure they fall within the valid class range
        segmentation_map_2d = np.clip(segmentation_map_2d, 0, len(self.id2color))

        resized_segmentation_map = self.resize_transform(Image.fromarray(segmentation_map_2d))

        # -----------------------------------------------------------------------------
        
        # Convert to NumPy and return the encoded inputs
        image_np = np.array(image)

        # comment if computing frames per second ------------------------------------------
        resized_segmentation_map = np.array(resized_segmentation_map)
        mask_tensor = torch.from_numpy(resized_segmentation_map).long()
        original_labels = torch.from_numpy(segmentation_map_2d).long()
       # -----------------------------------------------------------------------------

        # encoded_inputs = self.extractor(image, resized_segmentation_map, return_tensors="pt")
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()

        # comment if computing frames per seoncd ---------------------------
        return image_tensor, mask_tensor, original_labels
    
        # if running frames per second
        # return image_tensor
