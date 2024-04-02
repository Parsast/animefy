import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

class AnimeDataset(Dataset):
    def __init__(self, image_dir, image_size):
        self.image_dir = image_dir
        self.image_size = image_size
        self.paths = self.get_image_paths()
        

    def get_image_paths(self):
        paths = []
        for path in os.listdir(self.image_dir):
            if path.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png']:
                continue
            full_path = os.path.join(self.image_dir, path)
            if os.path.isfile(full_path):
                paths.append(full_path)
        return paths

    def __len__(self):
        return len(self.paths)

    def read_image(self, img_path):
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if 'style' in img_path or 'smooth' in img_path:
            image2 = np.zeros(image1.shape).astype(np.float32)  # Style/Smoothed
        else:
            seg_path = img_path.replace('train_photo', "seg_train_5-0.8-50")
            image2 = cv2.imread(seg_path)
            if image2 is None:
                raise FileNotFoundError(f"Could not read segmented image: {seg_path}")
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)  # Segmented 

        # Preprocessing 
        image1 = cv2.resize(image1, (self.image_size,self.image_size)) / 127.5 - 1.0 
        image2 = cv2.resize(image2, (self.image_size, self.image_size)) / 127.5 - 1.0 

        # Convert to PyTorch tensors
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()  
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()  

        return image1, image2 

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image1, image2 = self.read_image(img_path)
        return image1, image2

def get_dataloader(image_dir, image_size, batch_size, num_workers=0):
    dataset = AnimeDataset(image_dir=image_dir, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader


# if __name__ == '__main__':
#     # Test the data loader
#     real_image_dir = '/Users/parsa/codes/animefy/dataset/Hayao/smooth'
#     image_size = (256, 256)
#     batch_size = 8
#     num_workers = 0

#     real_image_loader = get_dataloader(real_image_dir, image_size, batch_size, num_workers)

#     # Print the number of batches
#     print(len(real_image_loader))

#     # Iterate over the data loader
#     for i, (real_images, anime_images) in enumerate(real_image_loader):
#         print(f'Batch {i+1}:')
#         print(f'Real images shape: {real_images.shape}')
#         print(f'Anime images shape: {anime_images.shape}')
#         break