import torch
import albumentations as albm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MPS_DEVICE = "mps"

DEVICE = MPS_DEVICE if torch.cuda.is_available() == False else "cuda"

# print(f"Device being used is {DEVICE}")

IMG_SIZE = 128
TRAIN_DIR = "/Users/mfonseca/Documents/Spring_23/local_STAT598_code/Project/v3_cloned_from_git/pytorch-CycleGAN-and-pix2pix/implementation_v4/dataset/mars/train"
SAVE_DIR = "/Users/mfonseca/Documents/Spring_23/local_STAT598_code/Project/v3_cloned_from_git/pytorch-CycleGAN-and-pix2pix/implementation_v4/saved_images"
VAL_DIR = "data/val"
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
LAMBDA_IDENTITY = 0.0               #set to zero initially.
LAMBDA_CYCLE = 10
NUM_WORKERS = 4                     #4
NUM_EPOCHS = 200                    #200
LOAD_MODEL = False
SAVE_MODEL = True

CHECKPOINT_GEN_A = "genA.pth.tar"
CHECKPOINT_GEN_B = "genB.pth.tar"
CHECKPOINT_DISC_A = "discA.pth.tar"
CHECKPOINT_DISC_B = "discB.pth.tar"