import zipfile
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pydicom import dcmread
import cv2
from pathlib import Path
import torch.nn as nn
from tqdm import tqdm
import albumentations as A # image augmentation library
import torch.optim as optim # optimizer
import torchvision.transforms.functional as tf
import copy
from sklearn.model_selection import GroupKFold

LIVER_INDICES = {1:  (27 , 124),
                 2:  (44 , 157),
                 3:  (61 , 182),
                 4:  (19 , 85 ),
                 5:  (19 , 137),
                 6:  (37 , 132),
                 7:  (48 , 145),
                 8:  (6  , 121),
                 9:  (11 , 99 ),
                 10: (22 , 120),
                 11: (33 , 128),
                 12: (9  , 247),
                 13: (26 , 115),
                 14: (4  , 107),
                 15: (6  , 120),
                 16: (39 , 151),
                 17: (1  , 113),
                 18: (12 , 70 ),
                 19: (28 , 68 ),
                 20: (126, 210),
            }

def unzip(zip_path, extract_path):
    """
    Extracts the compressed zip folder zip_path
    in the extract_path. If the extract_path folder
    doesn't exist already, it is created before the
    extraction.
    """
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"{zip_path} is extracted to: {extract_path}")
    return


def dicom_preprocess(path, min_hu=-100, max_hu=400):
    """
    Performs preprocessing on the DICOM image located at path.
    """
    dcm_file = dcmread(path)

    pixel_data = dcm_file.pixel_array.astype(np.float32)

    # HU conversion
    slope = getattr(dcm_file, "RescaleSlope", 1.0)
    intercept = getattr(dcm_file, "RescaleIntercept", 0.0)
    pixel_hu = pixel_data * slope + intercept

    # HU clipping, d'après Halder et al. il faut qu'il soit entre [-200,200] pour la tâche de la segmentation du foie
    pixel_clipped = np.clip(pixel_hu, min_hu, max_hu)
    return pixel_clipped

class LiverCTDataset(Dataset):

    def __init__(self, images: list, masks: list, transform=None, showHisto=False, dicom=True, min_hu=-100, max_hu=400):
        self.images = images
        self.masks = masks

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

        # print("Loading the following image, mask")
        # print(image_ID)
        # print(mask_ID)

        if self.dicom:
            image = dicom_preprocess(image_ID, min_hu=self.min_hu, max_hu=self.max_hu)
            image = (image - self.min_hu)/(self.max_hu - self.min_hu)
            image = np.expand_dims(image, -1)
            image = image.astype(np.float32)

            mask = dcmread(mask_ID).pixel_array
            mask = (mask > 0).astype(np.float32)
            mask = np.expand_dims(mask, -1)
            mask = mask.astype(np.float32)

        else:
            image = cv2.imread(image_ID, cv2.IMREAD_GRAYSCALE)
            image = image/255.
            image = np.expand_dims(image, -1)
            image = image.astype(np.float32)

            mask = cv2.imread(mask_ID, cv2.IMREAD_GRAYSCALE)
            mask = mask/255.
            mask = np.expand_dims(mask, -1)
            mask = mask.astype(np.float32)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']

        image, mask = torch.from_numpy(image).type(self.images_dtype), torch.from_numpy(mask).type(self.masks_dtype)
        image, mask = image.permute(2, 0, 1), mask.permute(2, 0, 1)

        if self.showHisto:
            plt.title(f"histogram for {os.path.splitext(image_ID)[0]}:")
            plt.hist(image.flatten())
            plt.xticks([-1000, -500, -200, 0, 200, 500, 1000])
            plt.show()

        return image, mask
    
# Load all the liver present dicom pathes from the dataset -------------------------------------------
def load_all_liver_appearances(base_path:Path, num_patients=[]):
    global LIVER_INDICES
    images, masks, patient_id = [], [], []

    to_load = LIVER_INDICES.keys() if num_patients == [] else num_patients
    
    for num_patient in to_load:
        first, last = LIVER_INDICES[num_patient]
        for ind in range(first, last+1):
            images.append(base_path/f"3Dircadb1/3Dircadb1.{num_patient}/PATIENT_DICOM/PATIENT_DICOM/image_{ind}")
            masks.append(base_path/f"3Dircadb1/3Dircadb1.{num_patient}/MASKS_DICOM/MASKS_DICOM/liver/image_{ind}")
            patient_id.append(num_patient)
            
    return images, masks, patient_id

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    

class Encoder(nn.Module):

    def __init__(self, in_channels=1, channels=(64, 128, 256, 512)):
        # in_channel=1 for grayscale, 3 for RGB
        super(Encoder, self).__init__()
        self.down = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for channel in channels:
            self.down.append(DoubleConv(in_channels, channel))
            in_channels = channel

        self.bottleneck = DoubleConv(channels[-1], channels[-1]*2)

    def forward(self, x):
        skip_connections = []

        for down in self.down:
            x = down(x)
            # skip connections store the feature maps from each level before they get downsampled
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        return x, skip_connections
    
class Decoder(nn.Module):

    def __init__(self, out_channel=1, channels=(512, 256, 128, 64)):
        super(Decoder, self).__init__()

        self.up = nn.ModuleList()

        for channel in channels:
            # upsampling convolution
            self.up.append(
                nn.ConvTranspose2d(in_channels=channel * 2, out_channels=channel, kernel_size=2, stride=2)
            )
            self.up.append(DoubleConv(channel * 2, channel))

        self.final_conv = nn.Conv2d(channels[-1], out_channel, kernel_size=1)

    def forward(self, x):
        out, skip_connections = x

        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.up), 2):
            out = self.up[i](out)
            skip_connection = skip_connections[i//2] # can't do i-1 because of index 0

            # avoids problems arising from the difference of size
            if out.shape != skip_connection:
                out = tf.resize(out, size=skip_connection.shape[2:])

            concat = torch.cat((skip_connection, out), dim=1)

            out = self.up[i+1](concat)

        return self.final_conv(out)
    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def dice_metric(pred, gt_mask, eps=1): # plus haut => meilleur
    pred = torch.sigmoid(pred)
    intersection = (pred * gt_mask).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + gt_mask.sum(dim=(2, 3))
    union = torch.where(union == 0, union + eps, union)
    dice_coeff = (2. * intersection) / union
    return dice_coeff.mean() # moyenne sur le batch

def dice_loss(pred, gt_mask, eps=1): # plus bas => meilleur
    dice_coeff = dice_metric(pred=pred, gt_mask=gt_mask, eps=eps)
    return 1 - dice_coeff


def main():
    print("start the process")
    base_path = Path("/users/Etu0/28708160")
    
    """
    print("started unzipping")
    unzip(zip_path=base_path/"3Dircadb1.zip", extract_path=base_path)
    for i in range(1, 21):
        current = f"3Dircadb1/3Dircadb1.{i}"
        unzip(zip_path=base_path/current/"MASKS_DICOM.zip", extract_path=base_path/current/"MASKS_DICOM")
        #convert_folder(dicom_dir=base_path/current/"MASKS_DICOM/MASKS_DICOM/liver", output_dir=base_path/current/"MASKS_DICOM"/"converted", patient_num=i, jpg=False, verbose=False, preprocess=False)
        unzip(zip_path=base_path/current/"PATIENT_DICOM.zip", extract_path=base_path/current/"PATIENT_DICOM")
    """
    
    # Hyperparameters
    LEARNING_RATE = 1e-4 # should conduct an experiment to determine the best value for this task
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)
    BATCH_SIZE = 4 # Rule of thumb: start batch_size = 4 or 8 and increase it until you run out of GPU memory
    NUM_EPOCHS = 100
    NUM_WORKERS = 4 #  Rule of thumb: num_workers = number of CPU cores // 2, in my case 10 cores
    IMG_H = 572
    IMG_W = 572

    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Image height and width: {IMG_H}, {IMG_W}")

    # train_images, train_masks, train_patient_id = load_all_liver_appearances(base_path=base_path, num_patients=list(range(1,19)))
    # test_images, test_masks, test_patient_id = load_all_liver_appearances(base_path=base_path, num_patients=list(range(19,21)))
    images, masks, patient_id = load_all_liver_appearances(base_path=base_path)
    p = 0.95

    # cross validation
    kf = GroupKFold(n_splits=5)
    for fold, (train_ind, test_ind) in enumerate(kf.split(images, masks, groups=patient_id)):
        print(f"fold {fold}")
        train_images = [images[i] for i in train_ind]
        train_masks = [masks[i] for i in train_ind]
        test_images = [images[i] for i in test_ind]
        test_masks = [masks[i] for i in test_ind]

        training_dataset = LiverCTDataset(images=train_images,
                                masks=train_masks,
                                transform=A.Compose([
                                        A.Affine(
                                            scale=(0.9, 1.1),
                                            translate_percent={"x": 0.0625, "y": 0.0625},
                                            rotate=(-45, 45),
                                            shear={"x": (-5, 5), "y": (-5, 5)},
                                            p=p
                                        ),
                                        A.RandomBrightnessContrast(p=p, brightness_limit=0.15, contrast_limit=0.15),
                                        A.Resize(IMG_H, IMG_W),
                                    ]),
                                showHisto=False)

        training_dataloader = DataLoader(dataset=training_dataset,
                                            batch_size=BATCH_SIZE,
                                            num_workers=NUM_WORKERS,
                                            shuffle=True)

        testing_dataset = LiverCTDataset(images=test_images,
                                masks=test_masks,
                                transform=A.Compose([
                                            A.Resize(IMG_H, IMG_W)
                                            ]),
                                showHisto=False)

        testing_dataloader = DataLoader(dataset=testing_dataset,
                                            batch_size=BATCH_SIZE,
                                            num_workers=NUM_WORKERS,
                                            shuffle=False)

        assert len(training_dataset) + len(testing_dataset) == 2073

        print("Loading first batch...")
        next(iter(training_dataloader))
        next(iter(testing_dataloader))
        print("First batch loaded.")

        # Checkpoint file path
        checkpoint_path = base_path/f'checkpoint_fold_{fold}.pth'
        start_epoch = 0

        model = UNet().to(device=DEVICE)
        optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE) # SGD?
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=3)

        # for early stopping
        best_loss = float('inf')
        patience = 10
        best_weights = None

        # initialization of the training process
        if os.path.exists(checkpoint_path):

            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            best_loss = checkpoint['best_loss']
            patience = checkpoint['patience']
            start_epoch = checkpoint['epoch'] + 1

            print(f"Resumed from epoch {start_epoch}")
            
        loss_func = dice_loss
        metric_func = dice_metric

        train_loss_all, test_loss_all = [], [] # mean loss per EPOCH
        train_metric_all, test_metric_all = [], []

        for ep in tqdm(range(start_epoch, NUM_EPOCHS)):
            #print("epoch")
            model.train()
            #print("train")
            step = 0
            current_train_loss = 0
            current_train_metric = 0

            # training loop
            for i, batch in enumerate(training_dataloader):
                #print(f"training loop num {i}")
                image_tmp, mask_tmp = batch
                image, mask = image_tmp.to(device=DEVICE), mask_tmp.to(device=DEVICE)

                optimizer.zero_grad()

                prediction = model(image)

                loss = loss_func(prediction, mask)
                loss.backward()
                optimizer.step()
                metric = metric_func(prediction, mask)

                current_train_loss += loss.item()
                current_train_metric += metric.item()

                step += 1
            train_loss_all.append(current_train_loss/step) # step = len(training_dataloader)?
            train_metric_all.append(current_train_metric/step)

            # test loop
            #print("after train")
            model.eval()
            #print("eval")
            step = 0
            current_test_loss = 0
            current_test_metric = 0

            with torch.no_grad():
                for i, batch in enumerate(testing_dataloader):
                    #print(f"testing loop num {i}")
                    image_tmp, mask_tmp = batch
                    image, mask = image_tmp.to(device=DEVICE), mask_tmp.to(device=DEVICE)

                    prediction = model(image)

                    loss = loss_func(prediction, mask)
                    metric = dice_metric(prediction, mask)

                    current_test_loss += loss.item()
                    current_test_metric += metric.item()

                    step += 1

                test_loss_all.append(current_test_loss/step)
                test_metric_all.append(current_test_metric/step)

            print("\n", "#"*50)
            print(f"[Epoch {ep+1}] Train Loss: {train_loss_all[-1]:}, Train Dice: {train_metric_all[-1]} | Test Loss: {test_loss_all[-1]}, Test Dice: {test_metric_all[-1]}")
            print("\n\n")

            # Early stopping, source https://medium.com/@vrunda.bhattbhatt/a-step-by-step-guide-to-early-stopping-in-tensorflow-and-pytorch-59c1e3d0e376
            if test_loss_all[-1] < best_loss:
                best_loss = test_loss_all[-1]
                best_weights = copy.deepcopy(model.state_dict())
                patience = 10 # resets the patience counter after improvement
            else:
                patience -= 1
                if patience == 0:
                    print(f"Early stopping triggered at epoch {ep+1}")
                    break

            scheduler.step(test_loss_all[-1])
            print(f"Learning Rate: {scheduler.get_last_lr()}")

            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'patience': patience,
            }, checkpoint_path)

        if best_weights is not None:   
            model.load_state_dict(best_weights) # reloads the golden checkpoint
            torch.save(best_weights, f"best_model_fold_{fold}.pth")
        else:
            print(f"best_weights is None at epoch {ep+1}")

if __name__ == "__main__":
    main()