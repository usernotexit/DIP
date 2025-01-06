import torch
from torch.utils.data import Dataset
import cv2
import os

class DatasetEdge2Shoes(Dataset):
    def __init__(self, dataset_folder, nData=2000):
        """
        Args:
            list_file (string): Path to the txt file with image filenames.
        """
        # Read the list of image filenames
        self.dataset_folder = dataset_folder
        self.nData = nData
        
    def __len__(self):
        # Return the total number of images
        return self.nData
    
    def __getitem__(self, idx):
        # Get the image filename
        img_name = os.path.join(self.dataset_folder, str(idx+1)+'_AB.jpg')
        
        img_color_semantic = cv2.imread(img_name)
        if img_color_semantic is None:
            raise ValueError(f"Unable to read image at {img_name}")

        # Convert the image to a PyTorch tensor
        img = torch.from_numpy(img_color_semantic).permute(2, 0, 1).float()/255.0 * 2.0 -1.0
        img_rgb = img[:,:,256:]
        img_edge = img[:,:,:256]
        return img_rgb, img_edge
