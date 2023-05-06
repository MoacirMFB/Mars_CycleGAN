import numpy as np
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os
import albumentations as al
import config


class Dataset_AB(Dataset):
    # _init_ Receives training root paths for images from both domains
    
    def __init__(self, root_domainA, root_domainB):  
        self.root_A = root_domainA;  self.root_B = root_domainB
        #define a series of transformations for the images
        self.transform_image = al.Compose([al.Resize(width=config.IMG_SIZE, height=config.IMG_SIZE),al.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],max_pixel_value = 255), ToTensorV2()],additional_targets = {"imageB":"imageA"})  

        #Read only image files
        self.imagesA = [file for file in os.listdir(root_domainA) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        self.imagesB = [file for file in os.listdir(root_domainB) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]

        # Length of the image datasets is not the same
        self.length_dataset = max(len(self.imagesA), len(self.imagesB))  # Check longest set of images

        self.imagesA_len = len(self.imagesA)
        self.imagesB_len = len(self.imagesB)

    def __getitem__(self, index):
        # Index could be greater, so we take modulus to avoid index errors when accessing images
        img_A = self.imagesA[index % self.imagesA_len]
        img_B = self.imagesB[index % self.imagesB_len]
        imgA_path = os.path.join(self.root_A, img_A)
        imgB_path = os.path.join(self.root_B, img_B)

        # Convert to RGB and numpy array for albumentations usage
        img_A = np.array(Image.open(imgA_path).convert("RGB"))
        img_B = np.array(Image.open(imgB_path).convert("RGB"))

        if self.transform_image:
            # Perform transformations on both domains of images
            img_A = self.transform_image(image=img_A)["image"]
            img_B = self.transform_image(image=img_B)["image"]

        return img_B, img_A
    
    def __len__(self):
        return self.length_dataset