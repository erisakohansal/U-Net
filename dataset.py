import torch
from torch.utils.data import Dataset
from pathlib import Path
from pydicom import dcmread
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

from utils import dicom_preprocess, load_all_liver_appearances

class LiverCTDataset(Dataset):
    # https://youtu.be/Sj-gIb0QiRM?si=zhdg12zIHM9E7ajD
    def __init__(self, images: list, masks: list, transform=None, showHisto=False, dicom=True, min_hu=-100, max_hu=400):
        self.images = images        # a list of images paths
        self.masks = masks          # a list of target paths

        self.transform = transform  

        self.images_dtype = torch.float32
        self.masks_dtype = torch.float32

        self.showHisto = showHisto
        self.dicom = dicom
        self.min_hu = min_hu
        self.max_hu = max_hu

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index: int):
        image_ID = self.images[index]
        mask_ID = self.masks[index]

        if self.dicom:
            image = dicom_preprocess(image_ID, min_hu=self.min_hu, max_hu=self.max_hu)
            image = (image - self.min_hu)/(self.max_hu - self.min_hu)
            image = np.expand_dims(image, -1)
            image = image.astype(np.float32)

            mask = dcmread(mask_ID).pixel_array
            mask = (mask > 0).astype(np.float32)  # Convertit tout pixel non nul en 1
            mask = np.expand_dims(mask, -1)
            mask = mask.astype(np.float32)

        else:
            image = cv2.imread(image_ID, cv2.IMREAD_GRAYSCALE)
            image = image/255.
            image = np.expand_dims(image, -1) # a single‐slice grayscale DICOM typically comes out [H, W] so you don't get [N, H, W] unless you explicitly insert the channel dimension
            # Note : Albumentations expects images in (H, W, C) format
            image = image.astype(np.float32)

            mask = cv2.imread(mask_ID, cv2.IMREAD_GRAYSCALE)
            mask = mask/255.
            mask = np.expand_dims(mask, -1)
            mask = mask.astype(np.float32)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']

        # making sure that we obtain a torch.tensor
        # usually we would have for inputs a torch.float32 and for 
        # target a torch.int64 to be used to create our dataloader
        image, mask = torch.from_numpy(image).type(self.images_dtype), torch.from_numpy(mask).type(self.masks_dtype)

        # Now shape: (1, H, W)
        image, mask = image.permute(2, 0, 1), mask.permute(2, 0, 1)   
        
        if self.showHisto:
            plt.title(f"histogram for {os.path.splitext(image_ID)[0]}:")
            plt.hist(image.flatten())
            plt.xticks([-1000, -500, -200, 0, 200, 500, 1000])
            plt.show()

        return image, mask
    
if __name__ == "__main__":
    base_path = Path("C:/Users/HP/Desktop/PIMA")

    import albumentations as A
    from torch.utils.data import DataLoader 

    p=0.95
    dicom_augmentation = A.Compose([
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent={"x": 0.0625, "y": 0.0625},
            rotate=(-45, 45),
            shear={"x": (-5, 5), "y": (-5, 5)},
            p=p
        ),
        A.RandomBrightnessContrast(p=p, brightness_limit=0.15, contrast_limit=0.15),
    ])

    # image = base_path/"3Dircadb1/3Dircadb1.2/PATIENT_DICOM/PATIENT_DICOM"
    # mask = base_path/"3Dircadb1/3Dircadb1.2/MASKS_DICOM/MASKS_DICOM/liver"
    # images = load_folder(image)
    # masks = load_folder(mask)

    images, masks = load_all_liver_appearances(base_path=base_path)

    training_dataset = LiverCTDataset(images=images, 
                            masks=masks,
                            transform=dicom_augmentation,
                            showHisto=False, 
                            dicom=True)

    training_dataloader = DataLoader(dataset=training_dataset, 
                                        batch_size=2, 
                                        shuffle=True)

    image, mask = next(iter(training_dataloader))

    print(f'image = shape: {image.shape}, type: {image.dtype}')
    print(f'image = min: {image.min()}, max: {image.max()}')
    print(f'mask = shape: {mask.shape}, class: {mask.unique()}, type: {mask.dtype}')
