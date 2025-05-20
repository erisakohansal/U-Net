import torch

LEARNING_RATE = 1e-4 # should conduct an experiment to determine the best value for this task
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4 # Rule of thumb: start batch_size = 4 or 8 and increase it until you run out of GPU memory
NUM_EPOCHS = 700
NUM_WORKERS = 4 #  Rule of thumb: num_workers = number of CPU cores // 2, in my case 10 cores
HU_MIN = -200
HU_MAX = 200
PIXDIM = (1., 1., 1.)
SPATIAL_SIZE = (128, 128, 128)
PATCH_SIZE = (64, 64, 64)
PROB = 0.8