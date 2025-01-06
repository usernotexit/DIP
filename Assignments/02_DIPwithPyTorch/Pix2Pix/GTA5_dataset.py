import torch
from torch.utils.data import Dataset
import cv2
import os

class GTA5Dataset(Dataset):
    def __init__(self, dataset_folder, list_file):
        """
        Args:
            list_file (string): Path to the txt file with image filenames.
        """
        # Read the list of image filenames
        list_file = os.path.join(dataset_folder, list_file)
        with open(list_file, 'r') as file:
            self.image_filenames = [line.strip() for line in file]
        self.imgs_folder = os.path.join(dataset_folder, 'images')
        self.labels_folder = os.path.join(dataset_folder, 'labels')
        
    def __len__(self):
        # Return the total number of images
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Get the image filename
        name = self.image_filenames[idx]
        img_name = os.path.join(self.imgs_folder, name)
        label_name = os.path.join(self.labels_folder, name)
        #print(img_name)

        img = cv2.imread(img_name)
        label = cv2.imread(label_name)

        # Convert the image to a PyTorch tensor
        image = torch.from_numpy(img).permute(2, 0, 1).float()/255.0 * 2.0 - 1.0
        label = torch.from_numpy(label).permute(2, 0, 1).float()/255.0 * 2.0 - 1.0
        return image, label