import torch
from torch.utils.data import Dataset
import cv2
import os

class FacadesDataset(Dataset):
    def __init__(self, dataset_folder, list_file):
        """
        Args:
            list_file (string): Path to the txt file with image filenames.
        """
        # Read the list of image filenames
        list_file = os.path.join(dataset_folder, list_file)
        self.dataset_folder = dataset_folder
        with open(list_file, 'r') as file:
            self.image_filenames = [line.strip()[2:] for line in file]
        
    def __len__(self):
        # Return the total number of images
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Get the image filename
        img_name = os.path.join(self.dataset_folder, self.image_filenames[idx])
        img_name = img_name.replace('\\','/')
        
        img_color_semantic = cv2.imread(img_name)
        if img_color_semantic is None:
            raise ValueError(f"Unable to read image at {img_name}")

        # Convert the image to a PyTorch tensor
        image = torch.from_numpy(img_color_semantic).permute(2, 0, 1).float()/255.0 * 2.0 -1.0
        image_rgb = image[:, :, :256]
        image_semantic = image[:, :, 256:]
        return image_rgb, image_semantic