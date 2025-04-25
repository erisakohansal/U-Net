import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from pydicom import dcmread
from pathlib import Path
from tqdm import tqdm
import albumentations as A # image augmentation library
import torch.optim as optim # optimizer
import copy

from unet import UNet2D
from dataset import LiverCTDataset
from loss import dice_loss_2d, dice_metric_2d
from utils import load_all_liver_appearances, unzip

def main():
    print("start the process")
    base_path = Path("/users/Etu0/28708160")
    
    # print("started unzipping")
    # unzip(zip_path=base_path/"3Dircadb1.zip", extract_path=base_path)
    # for i in range(1, 21):
    #     current = f"3Dircadb1/3Dircadb1.{i}"
    #     unzip(zip_path=base_path/current/"MASKS_DICOM.zip", extract_path=base_path/current/"MASKS_DICOM")
    #     #convert_folder(dicom_dir=base_path/current/"MASKS_DICOM/MASKS_DICOM/liver", output_dir=base_path/current/"MASKS_DICOM"/"converted", patient_num=i, jpg=False, verbose=False, preprocess=False)
    #     unzip(zip_path=base_path/current/"PATIENT_DICOM.zip", extract_path=base_path/current/"PATIENT_DICOM")
    

    # Hyperparameters
    LEARNING_RATE = 1e-4 # should conduct an experiment to determine the best value for this task
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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


    train_images, train_masks, train_patient_id = load_all_liver_appearances(base_path=base_path, num_patients=list(range(1,17)))
    val_images, val_masks, val_patient_id = load_all_liver_appearances(base_path=base_path, num_patients=list(range(17,19)))
    test_images, test_masks, test_patient_id = load_all_liver_appearances(base_path=base_path, num_patients=list(range(19,21))) 
    p = 0.95

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

    validation_dataset = LiverCTDataset(images=val_images,
                            masks=val_masks,
                            transform=A.Compose([
                                        A.Resize(IMG_H, IMG_W)
                                        ]),
                            showHisto=False)

    validation_dataloader = DataLoader(dataset=validation_dataset,
                                        batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORKERS,
                                        shuffle=False)
    
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

    assert len(training_dataset) + len(validation_dataset) == 2073

    print("Loading first batch...")
    next(iter(training_dataloader))
    next(iter(validation_dataloader))
    next(iter(testing_dataloader))
    print("First batch loaded.")

    # Checkpoint file path
    checkpoint_path = base_path/f'checkpoint.pth'
    start_epoch = 0

    model = UNet2D().to(device=DEVICE)
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
        
    loss_func = dice_loss_2d
    metric_func = dice_metric_2d

    train_loss_all, val_loss_all = [], [] # mean loss per EPOCH
    train_metric_all, val_metric_all = [], []

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

        # validation loop
        model.eval()
        step = 0
        current_val_loss = 0
        current_val_metric = 0

        with torch.no_grad():
            for i, batch in enumerate(validation_dataloader):
                image_tmp, mask_tmp = batch
                image, mask = image_tmp.to(device=DEVICE), mask_tmp.to(device=DEVICE)

                prediction = model(image)

                # vérifier les résultats
                if ep % 5 == 0:  # pour ne pas en avoir trop
                    pred_mask = torch.sigmoid(prediction[0, 0]).cpu().numpy()
                    gt_mask = mask[0, 0].cpu().numpy()
                    img = image[0, 0].cpu().numpy()
                    print("it should show something")

                    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                    axs[0].imshow(img, cmap='gray')
                    axs[0].set_title("Image d'entrée")
                    axs[1].imshow(gt_mask, cmap='gray')
                    axs[1].set_title("Masque réel")
                    axs[2].imshow(pred_mask > 0.5, cmap='gray')
                    axs[2].set_title("Masque prédit")
                    plt.suptitle(f"Epoch {ep+1}")
                    plt.tight_layout()
                    fig.savefig(f"./visu/epoch{ep+1}_i{i}.png")
                    plt.close(fig)

                loss = loss_func(prediction, mask)
                metric = metric_func(prediction, mask)

                current_val_loss += loss.item()
                current_val_metric += metric.item()

                step += 1

            val_loss_all.append(current_val_loss/step)
            val_metric_all.append(current_val_metric/step)

        print("\n", "#"*50)
        print(f"[Epoch {ep+1}] Train Dice: {train_metric_all[-1]} | Test Dice: {val_metric_all[-1]}")
        print("\n\n")

        # Early stopping, source https://medium.com/@vrunda.bhattbhatt/a-step-by-step-guide-to-early-stopping-in-tensorflow-and-pytorch-59c1e3d0e376
        if val_loss_all[-1] < best_loss:
            best_loss = val_loss_all[-1]
            best_weights = copy.deepcopy(model.state_dict())
            patience = 10 # resets the patience counter after improvement
        else:
            patience -= 1
            if patience == 0:
                print(f"Early stopping triggered at epoch {ep+1}")
                break

        scheduler.step(val_loss_all[-1])
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
        torch.save(best_weights, f"best_model.pth")
    else:
        print(f"best_weights is None at epoch {ep+1}")

    # testing loop
    step = 0
    current_test_loss = 0
    current_test_metric = 0
    with torch.no_grad():
        for i, batch in enumerate(testing_dataloader):
            image_tmp, mask_tmp = batch
            image, mask = image_tmp.to(device=DEVICE), mask_tmp.to(device=DEVICE)

            prediction = model(image)

            # vérifier les résultats
            pred_mask = torch.sigmoid(prediction[0, 0]).cpu().numpy()
            gt_mask = mask[0, 0].cpu().numpy()
            img = image[0, 0].cpu().numpy()
            print("it should show something")

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(img, cmap='gray')
            axs[0].set_title("Image d'entrée")
            axs[1].imshow(gt_mask, cmap='gray')
            axs[1].set_title("Masque réel")
            axs[2].imshow(pred_mask > 0.5, cmap='gray')
            axs[2].set_title("Masque prédit")
            plt.suptitle(f"Testing")
            plt.tight_layout()
            fig.savefig(f"./visu/testing_i{i}.png")
            plt.close(fig)

            loss = loss_func(prediction, mask)
            metric = metric_func(prediction, mask)

            current_test_loss += loss.item()
            current_test_metric += metric.item()

            step += 1

    print("\n", "#"*50)
    print(f"[Testing] Final Test Dice: {current_test_metric/step}")
    print("\n\n")

if __name__ == "__main__":
    main()