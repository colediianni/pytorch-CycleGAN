import os
import numpy

ROOT = os.getcwd()
OUTPUT_PATH = os.path.join(ROOT, 'output')
GLIOMA_MODEL_PATH = "/nobackup/datasets/gdrive/UoW_MQ_Glioma/metadata_csv_folder"
GAN_NORMALIZATION_MODEL_PATH = "/nobackup/datasets/gdrive/UoW_MQ_Glioma/GAN_model/tcga_G.pth"

# Used for setting random seed
SEED_DEFAULT = 123

# Default batch size
BATCH_SIZE_DEF = 128
EPOCHS_DEF = 10
# TODO: Add optimizer default settings

DATASETS = ['Train_GAN_Normalized', 'Test_GAN_Normalized', 'Train_Unnormalized', 'Test_Unnormalized', 'TCGA_Unnormalized', 'GAN_Generated', 'MUH_Unnormalized', 'BreakHis']
NORMALIZATION_METHODS = ["none", "gan", "..."]
OOD_DETECTION_METHODS = ["msp", "odin", "energy", "react", "gmm"]


FPR_MAX_PAUC = [0.01, 0.05, 0.1, 0.2]

# FPR threshold values for calculating TPR values.
# 0.1%, 0.5%, 1%, 5%, and 10%
FPR_THRESH = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]

RESNET_LAYERWISE_FILTER = [
    #input
    # 'conv1',
    # 'bn1',
    # 'relu',
    'maxpool',
    'layer1',
    'layer2',
    'layer3',
    # 'layer4',
    'avgpool'
     #output
]

# Normalizing Flows
LEARNING_RATE = 5e-3