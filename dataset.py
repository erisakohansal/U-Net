import torch
import monai
from monai.transforms import (
    Compose,
    LoadImage,
    LoadImaged,
    EnsureChannelFirst,
    EnsureChannelFirstd,
    Resize,
    Resized,
    ToTensor,
    ToTensord,
    Spacing,
    Spacingd,
    Orientation,
    Orientationd,
    ScaleIntensityRange,
    ScaleIntensityRanged,
    RandAffine,
    RandAffined, 
    RandAdjustContrast,
    RandAdjustContrastd, 
    Lambda,
    Lambdad,
)
from pathlib import Path
from pydicom import dcmread
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from utils import dicom_preprocess, load_liver_dicom, load_liver_nifti
from config import (DEVICE, BATCH_SIZE, NUM_WORKERS, HU_MIN, HU_MAX, PIXDIM, SPATIAL_SIZE, PATCH_SIZE)

class LiverCTDataset2Da(torch.utils.data.Dataset):
    def __init__(self, images: list, masks: list, transform=None, hu_min=HU_MIN, hu_max=HU_MAX):
        self.images = images      
        self.masks = masks         

        self.transform = transform  

        self.images_dtype = torch.float32
        self.masks_dtype = torch.float32

        self.hu_min = hu_min
        self.hu_max = hu_max

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index: int):
        image_ID = self.images[index]
        mask_ID = self.masks[index]

        image = dicom_preprocess(image_ID, min_hu=self.hu_min, max_hu=self.hu_max) # clipping
        image = (image - self.hu_min)/(self.hu_max - self.hu_min) # normalization
        image = np.expand_dims(image, -1)
        image = image.astype(np.float32)

        mask = dcmread(mask_ID).pixel_array
        mask = (mask > 0).astype(np.float32) 
        mask = np.expand_dims(mask, -1)
        mask = mask.astype(np.float32)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']

        image, mask = torch.from_numpy(image).type(self.images_dtype), torch.from_numpy(mask).type(self.masks_dtype) # ToTensor

        # Now shape: (1, H, W)
        image, mask = image.permute(2, 0, 1), mask.permute(2, 0, 1)   

        return image, mask
    
def LiverCTDataset2Db(images_path, masks_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, transform=None, hu_min=HU_MIN, hu_max=HU_MAX, shuffle=True): # add a file with global variables
    dataset_d = [{"image": image_file, "mask": mask_file} for image_file, mask_file in
                   zip(images_path, masks_path)]
    
    if transform is None:
        transform = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            ScaleIntensityRanged(keys=["image"], a_min=hu_min, a_max=hu_max, b_min=0.0, b_max=1.0, clip=True),
            Lambdad(keys=["mask"], func=lambda mask: (mask > 0).astype(np.float32)),
            ToTensord(keys=["image", "mask"]),
            Lambdad(keys=["image", "mask"], func=lambda x: x.permute(0, 3, 1, 2)),
        ]
    ) 
    
    dataset = monai.data.Dataset(data=dataset_d, transform=transform)
    dataloader = monai.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataset, dataloader

def LiverCTDataset3D(images_path, masks_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, transform=None, pixdim=PIXDIM, spatial_size=SPATIAL_SIZE, hu_min=HU_MIN, hu_max=HU_MAX, shuffle=True): 
    dataset_d = [{"image": image_file, "mask": mask_file} for image_file, mask_file in
                   zip(images_path, masks_path)]
    
    if transform is None:
        transform = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            Spacingd(keys=["image", "mask"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Resized(keys=["image", "mask"], spatial_size=spatial_size, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=hu_min, a_max=hu_max, b_min=0.0, b_max=1.0, clip=True),
            Lambdad(keys="mask", func=lambda mask: (mask > 0).astype(np.float32)),
            ToTensord(keys=["image", "mask"]),
            Lambdad(keys=["image", "mask"], func=lambda x: x.permute(0, 3, 1, 2)),
        ]
    ) 

    dataset = monai.data.Dataset(data=dataset_d, transform=transform)
    dataloader = monai.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataset, dataloader

def LiverCTDataset3DPatch(images_path, masks_path, device=DEVICE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, transform=None, pixdim=PIXDIM, spatial_size=SPATIAL_SIZE, patch_size=PATCH_SIZE,  hu_min=HU_MIN, hu_max=HU_MAX): # add a file with global variables

    dataset_d = [{"image": image_file, "mask": mask_file} for image_file, mask_file in
                   zip(images_path, masks_path)]
    
    if transform is None:

        transform = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            Spacingd(keys=["image", "mask"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Resized(keys=["image", "mask"], spatial_size=spatial_size, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=hu_min, a_max=hu_max, b_min=0.0, b_max=1.0, clip=True),
            Lambdad(keys="mask", func=lambda mask: (mask > 0).astype(np.float32)),
            ToTensord(keys=["image", "mask"]),
            Lambdad(keys=["image", "mask"], func=lambda x: x.permute(0, 3, 1, 2)),
        ]
    ) 

    dataset = monai.data.Dataset(data=dataset_d, transform=transform)
    patch_iter = monai.data.PatchIter(patch_size=patch_size, start_pos=(0, 0, 0))

    def image_mask_iter(x):
        for im, seg in zip(patch_iter(x["image"]), patch_iter(x["mask"])):
            yield ((im[0], seg[0]),)

    dataset = monai.data.GridPatchDataset(dataset, image_mask_iter, with_coordinates=False)
    dataloader = monai.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=device)
    return dataset, dataloader
    
if __name__ == "__main__":
    base_path = Path("C:/Users/HP/Desktop/PIMA")

    # test 2D
    # p=0.95
    # dicom_augmentation = A.Compose([
        
    #     A.RandomBrightnessContrast(p=p, brightness_limit=0.15, contrast_limit=0.15),

    # ])

    # images, masks, _ = load_liver_dicom(base_path=base_path)

    # training_dataset = LiverCTDataset2Da(images=images, 
    #                         masks=masks,
    #                         transform=dicom_augmentation)

    # training_dataloader = torch.utils.data.DataLoader(dataset=training_dataset, 
    #                                     batch_size=2, 
    #                                     shuffle=True)

    # for batch in training_dataloader:
    #     image, mask = batch


    #     print(f'image = shape: {image.shape}, type: {image.dtype}')
    #     print(f'image = min: {image.min()}, max: {image.max()}')
    #     print(f'mask = shape: {mask.shape}, class: {mask.unique()}, type: {mask.dtype}')
    #     plt.figure("Visualization train", (12, 6))
    #     plt.subplot(1, 2, 1)
    #     plt.title(f"image")
    #     plt.imshow(image[0, 0], cmap="gray")

    #     plt.subplot(1, 2, 2)
    #     plt.title(f"mask")
    #     plt.imshow(mask[0, 0], cmap="gray")
    #     plt.show()

    # test 3D

    # p = 0.9
    # transform_3d = Compose([
    #     LoadImaged(keys=["image", "mask"]),
    #     EnsureChannelFirstd(keys=["image", "mask"]),
    #     RandAffined(
    #         keys=["image", "mask"],
    #         prob=p,
    #         rotate_range=(np.pi / 4,),         
    #         shear_range=(0.087, 0.087),       
    #         scale_range=(0.1, 0.1),            
    #         translate_range=(0.0625, 0.0625),  
    #         mode=("bilinear", "nearest"),
    #         padding_mode="border"
    #     ),
    #     RandAdjustContrastd(keys=["image"], prob=p, gamma=(0.9, 1.1)),
    #     ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=400, b_min=0.0, b_max=1.0, clip=True),
    #     Lambdad(keys="mask", func=lambda mask: (mask > 0).astype(np.float32)),
    #     ToTensord(keys=["image", "mask"])
    # ])
    #images, masks = load_liver_nifti(base_path/"Volumes", num_patients=[4, 18, 20])
    # _, training_dataloader = LiverCTDataset2Db(images, masks, transform=transform_3d)

    # for batch in training_dataloader:
    #     image = batch["image"]
    #     mask = batch["mask"]

    #     print(f'image = shape: {image.shape}, type: {image.dtype}')
    #     print(f'image = min: {image.min()}, max: {image.max()}')
    #     print(f'mask = shape: {mask.shape}, class: {mask.unique()}, type: {mask.dtype}')
    #     plt.figure("Visualization train", (12, 6))
    #     plt.subplot(1, 2, 1)
    #     plt.title(f"image")
    #     plt.imshow(image[0, 0], cmap="gray")

    #     plt.subplot(1, 2, 2)
    #     plt.title(f"mask")
    #     plt.imshow(mask[0, 0], cmap="gray")
    #     plt.show()


    # for i, (image, mask) in enumerate(training_dataloader):
    #     print(f"Patch {i}, image shape: {image.shape}, mask shape: {mask.shape}")

    #     center = image.shape[2] // 2  # profondeur = D (axe 2 après permute)
    #     image_slice = image[0, 0, center].cpu()
    #     mask_slice = mask[0, 0, center].cpu()

    #     plt.figure(figsize=(5, 5))
    #     plt.imshow(image_slice, cmap='gray')
    #     plt.imshow(mask_slice, cmap='Reds', alpha=0.4)
    #     plt.title(f'Patch {i}')
    #     plt.show()

    # patches = list(training_dataloader)
    # total_patches = len(patches)
    # n_images = int(len(patches) / ((SPATIAL_SIZE[0]/PATCH_SIZE[0]) ** 3))
    # n_patches_per_image = total_patches // n_images
    # target_patch_index = 7
    # for i in range(n_images):
    #     index = i * n_patches_per_image + target_patch_index
    #     (image, mask) = patches[index]

    #     image = image.cpu()
    #     mask = mask.cpu()
    #     center_slice = image.shape[1] // 2  # axe D après permute

    #     image_slice = image[0,0,center_slice]
    #     mask_slice = mask[0,0, center_slice]

    #     plt.figure(figsize=(5, 5))
    #     plt.imshow(image_slice, cmap="gray")
    #     plt.imshow(mask_slice, cmap="Reds", alpha=0.4)
    #     plt.title(f"Image {i}, patch {target_patch_index}")
    #     plt.axis("off")
    #     plt.show()
    BATCH_SIZE = 1
    NUM_WORKERS = 0

    test_images, test_masks = load_liver_nifti(path=base_path/"Volumes", 
                                        num_patients=[4,18,20])

    _, testing_dataloader = LiverCTDataset3DPatch(device=DEVICE, images_path=test_images, 
                                                  masks_path=test_masks, batch_size=BATCH_SIZE, 
                                                  num_workers=NUM_WORKERS, pixdim=PIXDIM, spatial_size=SPATIAL_SIZE, 
                                                  patch_size=PATCH_SIZE, hu_min=HU_MIN, hu_max=HU_MAX)

    
    print("Loading first batch...")
    current_patch = 0
    current_image_index = 0
    target_patch_index = 7
    patches_per_image = int((SPATIAL_SIZE[0] / PATCH_SIZE[0]) ** 3)

    for i, (image, mask) in enumerate(testing_dataloader):  # ou ((image, mask), coord) selon ton setup

        if current_patch == target_patch_index:
            print(f"Image {current_image_index}, Patch {target_patch_index}")
            print(f'image = shape: {image.shape}, type: {image.dtype}')
            print(f'image = min: {image.min()}, max: {image.max()}')
            print(f'mask = shape: {mask.shape}, class: {mask.unique()}, type: {mask.dtype}')

            center_slice = image.shape[2] // 2  

            image_slice = image[0, 0, center_slice]
            mask_slice = mask[0, 0, center_slice]

            plt.figure("Visualization train", (5, 5))
            plt.imshow(image_slice, cmap="gray")
            plt.imshow(mask_slice, cmap="Reds", alpha=0.4)
            plt.show()

        current_patch += 1

        if current_patch % patches_per_image == 0:
            current_image_index += 1
            current_patch = 0
            
        
