import torch
from torch.utils.data import Dataset
from pathlib import Path
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

from utils import dicom_preprocess, load_folder

class LiverCTDataset(Dataset):
    # https://youtu.be/Sj-gIb0QiRM?si=zhdg12zIHM9E7ajD
    def __init__(self, inputs: list, masks: list, transform=None, showHisto=False):
        self.inputs = inputs        # a list of inputs paths
        self.masks = masks          # a list of target paths

        self.transform = transform  
        self.inputs_dtype = torch.float32
        self.masks_dtype = torch.long # =torch.int64
        self.showHisto = showHisto

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index: int):
        inputs_ID = self.inputs[index]
        masks_ID = self.masks[index]

        # Image
        image = cv2.imread(inputs_ID, cv2.IMREAD_GRAYSCALE)
        image = image/255.
        print(image.shape)
        image = np.expand_dims(image, -1) # a single‐slice grayscale DICOM typically comes out [H, W] so you don't get [N, H, W] unless you explicitly insert the channel dimension
        # Note : Albumentations expects images in (H, W, C) format
        print(image.shape)
        image = image.astype(np.float32)
        #image = torch.from_numpy(image)

        # Mask
        mask = cv2.imread(masks_ID, cv2.IMREAD_GRAYSCALE)
        mask = mask/255.
        print(mask.shape)
        mask = np.expand_dims(mask, -1)
        print(mask.shape)
        mask = mask.astype(np.float32)
        #mask = torch.from_numpy(mask)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']

        # making sure that we obtain a torch.tensor
        # usually we would have for inputs a torch.float32 and for 
        # target a torch.int64 to be used to create our dataloader
        image, mask = torch.from_numpy(image).type(self.inputs_dtype), torch.from_numpy(mask).type(self.masks_dtype)

        # Now shape: (1, H, W)
        image = image.permute(2, 0, 1)  
        mask  = mask.permute(2, 0, 1)   
        
        if self.showHisto:
            plt.title(f"histogram for {os.path.splitext(inputs_ID)[0]}:")
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
        A.OneOf([
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            A.Transpose(p=p),
            A.RandomRotate90(p=p),
            A.ShiftScaleRotate(p=p, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45)
        ], p=1),
        A.ElasticTransform(p=p, alpha=15, sigma=1, interpolation=cv2.INTER_NEAREST),
        A.RandomBrightnessContrast(p=p, brightness_limit=0.15, contrast_limit=0.15),
        A.PadIfNeeded(p=1, min_height=128, min_width=128, border_mode=cv2.BORDER_REFLECT)
    ])

    # exemple : lire les images du premier patient : image_27 et image_70
    input = base_path/"Data/Inputs"
    mask = base_path/"Data/Masks"

    inputs = load_folder(input)
    masks = load_folder(mask)

    training_dataset = LiverCTDataset(inputs=inputs, 
                            masks=masks,
                            transform=dicom_augmentation,
                            showHisto=False)

    training_dataloader = DataLoader(dataset=training_dataset, 
                                        batch_size=2, 
                                        shuffle=True)

    image, mask = next(iter(training_dataloader))

    print(f'image = shape: {image.shape}, type: {image.dtype}')
    print(f'image = min: {image.min()}, max: {image.max()}')
    print(f'mask = shape: {mask.shape}, class: {mask.unique()}, type: {mask.dtype}')
