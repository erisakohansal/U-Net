import os
import copy
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from config import (LEARNING_RATE, DEVICE, BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, PATCH_SIZE, PIXDIM, SPATIAL_SIZE)

# train.py
def train(model, optimizer, scheduler, checkpoint_path, loss_func, training_dataloader, validation_dataloader, patience=10, min_delta=0.001, load_checkpoint=False, train_2D=True, patch=False):

    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Workers: {NUM_WORKERS}")

    if not train_2D:
        print(f"Pixel Dimension: {PIXDIM}")
        print(f"Spatial Size: {SPATIAL_SIZE}")
        if patch: print(f"Patch Size: {PATCH_SIZE}")
        
    # for early stopping
    start_epoch = 0
    early_stopping = float('inf')
    best_loss = float('inf')
    best_weights = None

    # initialization of the training process
    if load_checkpoint and os.path.exists(checkpoint_path):

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        best_loss = checkpoint['best_loss']
        patience = checkpoint['patience']
        start_epoch = checkpoint['epoch'] + 1

        print(f"Resumed from epoch {start_epoch}")

    train_loss_all, val_loss_all = [], [] # mean loss per EPOCH
    train_metric_all, val_metric_all = [], []

    for ep in tqdm(range(start_epoch, NUM_EPOCHS)):
        model.train()
        step = 0
        current_train_loss = 0
        current_train_metric = 0

        # training loop
        for i, batch in enumerate(training_dataloader):
            image_tmp, mask_tmp = None, None
            if patch:
                image_tmp, mask_tmp = batch
            else: 
                image_tmp, mask_tmp = batch["image"], batch["mask"]
            image, mask = image_tmp.to(device=DEVICE), mask_tmp.to(device=DEVICE)

            optimizer.zero_grad()

            prediction = model(image)

            loss = loss_func(prediction, mask)
            loss.backward()
            optimizer.step()
            metric = 1 - loss.item()

            current_train_loss += loss.item()
            current_train_metric += metric

            step += 1
        train_loss_all.append(current_train_loss/step) # step = len(training_dataloader)?
        train_metric_all.append(current_train_metric/step)

        # validation loop
        model.eval() 
        step = 0
        current_val_loss = 0
        current_val_metric = 0
        target_patch_index = 7
        patches_per_image = int((SPATIAL_SIZE[0] / PATCH_SIZE[0]) ** 3)

        with torch.no_grad():
            for i, batch in enumerate(validation_dataloader):
                image_tmp, mask_tmp = None, None
                if patch:
                    image_tmp, mask_tmp = batch
                else: 
                    image_tmp, mask_tmp = batch["image"], batch["mask"]
                image, mask = image_tmp.to(device=DEVICE), mask_tmp.to(device=DEVICE)

                prediction = model(image)

                # vérifier les résultats
                if ep % 5 == 0:  # pour ne pas en avoir trop
                    pred_mask = torch.sigmoid(prediction[0, 0]).cpu().numpy()
                    gt_mask = mask[0, 0].cpu().numpy()
                    img = image[0, 0].cpu().numpy()

                    slice_ind = 0 if train_2D else pred_mask.shape[0]//2

                    if patch:
                        if current_patch == target_patch_index:
                            print(f"Image {current_image_index}, Patch {target_patch_index}")
                            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                            axs[0].imshow(img if train_2D else img[slice_ind], cmap='gray')
                            axs[0].set_title("Image d'entrée")
                            axs[1].imshow(gt_mask if train_2D else gt_mask[slice_ind], cmap='gray')
                            axs[1].set_title("Masque réel")
                            axs[2].imshow(pred_mask > 0.5 if train_2D else (pred_mask > 0.5)[slice_ind], cmap='gray')
                            axs[2].set_title("Masque prédit")
                            plt.suptitle(f"Epoch {ep+1}, batch {i+1}")
                            plt.tight_layout()
                            plt.show()

                        current_patch += 1

                        # Incrémente l'image index si on termine un groupe
                        if current_patch % patches_per_image == 0:
                            current_image_index += 1
                            current_patch = 0
                        
                    else:
                        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                        axs[0].imshow(img if train_2D else img[slice_ind], cmap='gray')
                        axs[0].set_title("Image d'entrée")
                        axs[1].imshow(gt_mask if train_2D else gt_mask[slice_ind], cmap='gray')
                        axs[1].set_title("Masque réel")
                        axs[2].imshow(pred_mask > 0.5 if train_2D else (pred_mask > 0.5)[slice_ind], cmap='gray')
                        axs[2].set_title("Masque prédit")
                        plt.suptitle(f"Epoch {ep+1}, batch {i+1}")
                        plt.tight_layout()
                        plt.show()

                loss = loss_func(prediction, mask)
                metric = 1 - loss.item()

                current_val_loss += loss.item()
                current_val_metric += metric

                step += 1

            val_loss_all.append(current_val_loss/step)
            val_metric_all.append(current_val_metric/step)

        print("\n", "#"*50)
        print(f"[Epoch {ep+1}] Train Dice: {train_metric_all[-1]} | Validation Dice: {val_metric_all[-1]}")

        # Early stopping, source https://medium.com/@vrunda.bhattbhatt/a-step-by-step-guide-to-early-stopping-in-tensorflow-and-pytorch-59c1e3d0e376
        if best_loss - val_loss_all[-1] > min_delta:
            best_loss = val_loss_all[-1]
            best_weights = copy.deepcopy(model.state_dict())
            patience = 10
        else:
            patience -= 1
            if patience == 0:
                early_stopping = ep+1
                print(f"Early stopping triggered at epoch {ep+1}")
                break    

        scheduler.step(val_loss_all[-1])
        print(f"Learning Rate: {scheduler.get_last_lr()}")
        print("\n\n")

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

    return early_stopping, train_metric_all, val_metric_all

def test(model, loss_func, testing_dataloader, test_2D=True, patch=False):
    # testing loop
    step = 0
    current_test_loss = 0
    current_test_metric = 0
    current_patch = 0
    current_image_index = 0
    target_patch_index = 7
    patches_per_image = int((SPATIAL_SIZE[0] / PATCH_SIZE[0]) ** 3)


    with torch.no_grad():
        for i, batch in enumerate(testing_dataloader):
            image_tmp, mask_tmp = None, None
            if patch:
                image_tmp, mask_tmp = batch
            else: 
                image_tmp, mask_tmp = batch["image"], batch["mask"]
            image, mask = image_tmp.to(device=DEVICE), mask_tmp.to(device=DEVICE)

            prediction = model(image)

            pred_mask = torch.sigmoid(prediction[0, 0]).cpu().numpy()
            gt_mask = mask[0, 0].cpu().numpy()
            img = image[0, 0].cpu().numpy()

            slice_ind = 0 if test_2D else pred_mask.shape[0]//2

            if patch:
                if current_patch == target_patch_index:
                    print(f"Image {current_image_index}, Patch {target_patch_index}")

                    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                    axs[0].imshow(img if test_2D else img[slice_ind], cmap='gray')
                    axs[0].set_title("Image d'entrée")
                    axs[1].imshow(gt_mask if test_2D else gt_mask[slice_ind], cmap='gray')
                    axs[1].set_title("Masque réel")
                    axs[2].imshow(pred_mask > 0.5 if test_2D else (pred_mask > 0.5)[slice_ind], cmap='gray')
                    axs[2].set_title("Masque prédit")
                    plt.suptitle(f"Image {current_image_index}, patch {target_patch_index}")
                    plt.tight_layout()
                    plt.show()

                current_patch += 1

                # Incrémente l'image index si on termine un groupe
                if current_patch % patches_per_image == 0:
                    current_image_index += 1
                    current_patch = 0

            else:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(img if test_2D else img[slice_ind], cmap='gray')
                axs[0].set_title("Image d'entrée")
                axs[1].imshow(gt_mask if test_2D else gt_mask[slice_ind], cmap='gray')
                axs[1].set_title("Masque réel")
                axs[2].imshow(pred_mask > 0.5 if test_2D else (pred_mask > 0.5)[slice_ind], cmap='gray')
                axs[2].set_title("Masque prédit")
                plt.suptitle(f"Testing")
                plt.tight_layout()
                plt.show()

            loss = loss_func(prediction, mask)
            metric = 1 - loss.item()

            current_test_loss += loss.item()
            current_test_metric += metric

            step += 1

    print("\n", "#"*50)
    print(f"[Testing] Final Test Dice: {current_test_metric/step}")
    print("\n\n")

    return current_test_metric/step

def show_dice(early_stopping, train_metric_all, val_metric_all, current_test_metric):
    plt.plot(list(range(early_stopping)), train_metric_all, 'r-', label='Train Dice')
    plt.plot(list(range(early_stopping)), val_metric_all, 'g-', label='Validation Dice')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.title(f"Training and Validation Dice over Epochs (final test Dice: {current_test_metric:.2f})")
    plt.show()