import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

# This class generates the dataset
class RetinaDataset(Dataset):
    
    def __init__(self, data_dir, split, transform=None):  
        self.data_dir = data_dir                           # Initialize the directory for data
        self.transform = transform                         # Initialize the transformation parameter
        self.image_paths = []                              # Initialize the list to store paths to the images
        self.labels = []                                   # Initialize the list to store the label vector of each sample
        
        csv_path = os.path.join(data_dir, './Groundtruths', f'{split}_data.csv') # stores the path to the label csv file
        data = pd.read_csv(csv_path)                                           # data stores the label dataframe
        
        for idx, row in data.iterrows():   # idx stores the column index and row stores the row for each sample
            
            if os.path.exists(os.path.join(data_dir, 'images', f'{row["ID"]}.tif')): # check the extension of the image to add to the image path
                image_path = os.path.join(data_dir, 'images', f'{row["ID"]}.tif') 
            else: 
                image_path = os.path.join(data_dir, 'images', f'{row["ID"]}.png')
                
            self.image_paths.append(image_path)                                      # Appending paths to image_paths
            self.labels.append(row[1:].tolist())                                     # Appending label vectors to labels
    
    def __len__(self):                                                        
        return len(self.image_paths)
    
    # This method returns a sample image with applied transformation and the PyTorch tensor for the labels
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)                                     # To open an image from the path
        
        if self.transform:                                                 # Applying transformation to image
            image = self.transform(image)
            
        label = torch.tensor(self.labels[idx], dtype=torch.float32)        # Generating label tensor
        
        return image, label                                                