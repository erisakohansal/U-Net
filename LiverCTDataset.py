import torch
from pydicom import dcmread
from torch.utils import data
from pathlib import Path

#from Config import preprocess
from Loaders import * 

# to isntall torch : pip3 install torch torchvision torchaudio
class LiverCTDataset(data.Dataset):
    def __init__(self, inputs: list, masks: list, transform=None):
        self.inputs = inputs        # a list of inputs paths
        self.masks = masks          # a list of target paths

        self.transform = transform  # it's a function to process the data
        
        # CHECK THIS PART????
        self.inputs_dtype = torch.float32
        self.masks_dtype = torch.long # =torch.int64

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index: int):
        # Selects the sample
        inputs_ID = self.inputs[index]
        target_ID = self.masks[index]

        # Loads inputs and target
        x, y = dcmread(inputs_ID).pixel_array, dcmread(target_ID).pixel_array

        # Preprocessing + data augmentation = transform
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting, making sure that we obtain a torch.tensor
        # usually we would have for inputs a torch.float32 and for 
        # target a torch.int64, to be used to create our dataloader
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.masks_dtype)

        # Add channel dimension
        #  a single‐slice grayscale DICOM typically comes out [H, W], so you get [N, H, W] unless you explicitly insert the channel dimension.
        x = x.unsqueeze(0)  # Now shape is [1, H, W]

        return x, y
    
if __name__ == "__main__":
    base_path = Path("C:/Users/HP/Desktop/PIMA/3Dircadb1")
    inputs, _, masks = load_all_dicom(base_path=base_path)

    training_dataset = LiverCTDataset(inputs=inputs, 
                            masks=masks,
                            transform=None)

    training_dataloader = data.DataLoader(dataset=training_dataset, 
                                        batch_size=2823, 
                                        shuffle=True) # shuffling is usually done on training data and not on test/validation sets
    
    x, y = next(iter(training_dataloader))
    
    print(f'x = shape: {x.shape}; type: {x.dtype}')
    print(f'x = min: {x.min()}; max: {x.max()}')
    print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')
