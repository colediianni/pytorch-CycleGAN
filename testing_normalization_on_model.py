# TODO use saved normalization model to check accuracy of classification model

from __future__ import absolute_import, division, print_function
import argparse
import torch
from tqdm import tqdm
from torchvision import datasets, transforms
import os

from PIL import Image
import cv2
import sys
# from pympler.asizeof import asizeof
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from helpers.constants import (
    ROOT,
    OUTPUT_PATH,
    GLIOMA_MODEL_PATH,
    SEED_DEFAULT,
    BATCH_SIZE_DEF,
    EPOCHS_DEF,
    NORMALIZATION_METHODS,
    OOD_DETECTION_METHODS,
    DATASETS
)
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from data.base_dataset import BaseDataset, get_transform
from helpers.utils import (
    load_model_checkpoint,
    get_num_jobs
    #     extract_layer_embeddings,
    #     get_activations
)
from datasets.glioma_data_loader import (
    #     test_model,
    WSIDataset,
    My_Normalize
    #     patent_class
)
# from helpers.score import (
#     get_score
# )
try:
    import cPickle as pickle
except:
    import pickle


# optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def main():
    
    opt = TestOptions().parse()  # get test options

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    use_cuda = not opt.no_cuda and torch.cuda.is_available()

    torch.manual_seed(opt.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    n_jobs = get_num_jobs(opt.n_jobs)
    
    # Load pretrained model with x normalization
    if opt.model_type == 'unnormalized':
        model_name = "ResNet_Classifier_original.pt"
        training_data = "TCGA_Unnormalized"
    elif opt.model_type == 'gan-normalized':
        model_name = "resnet50_cnn.pt"
        training_data = "Train_GAN_Normalized"
    else:
        raise ValueError("{} is not an available classification model".format(opt.model_type))

    model = load_model_checkpoint(model_name, device)
    model.eval()
    num_classes = 2

    wsi_dataset = WSIDataset(GLIOMA_MODEL_PATH)

    test_data_loader = wsi_dataset.Obtain_loader(
        opt.dataset, opt.batch_size, n_jobs)
    
    total_predictions = 0
    correct_predictions = 0
    for image_data in tqdm(test_data_loader):
        data = image_data['data'].to(device=device, dtype=torch.float)
        # print(data.max(), data.min()) # tensor(2.6400, device='cuda:1') tensor(-2.0357, device='cuda:1')
        # plt.imshow(data.cpu().detach().numpy()[0].transpose((1, 2, 0)))
        # plt.savefig("/nobackup/cole/pytorch-CycleGAN/testing_image.jpg")
        real = image_data['target'].flatten()
        with torch.no_grad():
            logits = model(data.to(device))
        prediction = np.argmax(torch.Tensor.cpu(logits), axis=1)
        correct_predictions += sum(real == prediction)
        total_predictions += len(prediction)
    # print(data.cpu().detach().numpy()[0].transpose((1, 2, 0)).max(), data.cpu().detach().numpy()[0].transpose((1, 2, 0)).min())
    print("model accuracy on {} without normalization: {}".format(opt.dataset, correct_predictions/total_predictions))
    
    norm_model = create_model(opt)      # create a model given opt.model and other options
    norm_model.setup(opt)               # regular setup: load and print networks; create s
    norm_model = norm_model.netG_B    
    
    total_predictions = 0
    correct_predictions = 0
    cycle_transform = get_transform(opt, grayscale=False, convert=False)
    classifier_transform = My_Normalize()
    for image_data in tqdm(test_data_loader):
        data_path = image_data['path'][0]
        # print(data_path)
        # image = Image.open(data_path).convert('RGB').resize((256, 256))
        
        image = cv2.imread(data_path, cv2.COLOR_BGR2RGB)
        image= cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        # print(image.size)
        # test_image = transforms.Compose([transforms.ToTensor()])(image)
        # print(test_image.shape)
        # print(test_image.max(), test_image.min())
        # print(image2.shape)
        # print(image2.max(), image2.min())
        # print(cycle_transform(image).to(device).max(), cycle_transform(image).to(device).min()) # tensor(1., device='cuda:1') tensor(0.0706, device='cuda:1')
        image = torch.unsqueeze(cycle_transform(image).to(device), dim=0)
        
        # visuals = cycle_transform(image).to(device)
        
        # plt.imshow(image.cpu().detach().numpy()[0].transpose((1, 2, 0)))
        # plt.savefig("/nobackup/cole/pytorch-CycleGAN/testing_image.jpg")
        # norm_model.set_input({'A': image, 'B': image, 'A_paths': "nope", 'B_paths': "nope"})  # unpack data from data loader
        # norm_model.test()  # run inference
        # visuals = norm_model.get_current_visuals()  # get image results. Stored as real_X, fake_Y, rec_X (reconstructed)

        with torch.no_grad():
            visuals = norm_model(image)

        # visuals = model.get_current_visuals()  # get image results. Stored as real_X, fake_Y, rec_X (reconstructed)
        # images = visuals["fake_A"]
        visuals = torch.squeeze(visuals)
        # print(visuals.max(), visuals.min(), visuals.mean())
        visuals = 255 * visuals # Unnormalize because current image is between -1 and 1
        # print(visuals.max(), visuals.min())
        visuals = torch.clamp(visuals, min=0, max=255)
        
        # plt.imshow(visuals.cpu().detach().numpy().transpose((1, 2, 0))/255)
        # plt.savefig("/nobackup/cole/pytorch-CycleGAN/testing_image.jpg")
        
        # print(visuals.shape)
        # print(visuals.max(), visuals.min())
        
        visuals = classifier_transform({'data': visuals.cpu().detach().numpy().transpose((1, 2, 0)), 'target': 0, 'p_id': "none", 'path': "none"})['data']
        # visuals = visuals.transpose((2, 0, 1))
        # plt.imshow(visuals.numpy().transpose((1, 2, 0)))
        # plt.savefig("/nobackup/cole/pytorch-CycleGAN/testing_image.jpg")
        visuals = torch.unsqueeze(visuals, dim=0)
        
        # print(visuals.shape)
        # print(visuals.max(), visuals.min()) # tensor(2.5379) tensor(-2.2177)
        
        real = image_data['target'].flatten()
        with torch.no_grad():
            # logits = model(torch.tensor(visuals).to(device))
            logits = model(visuals.to(device))
        prediction = np.argmax(torch.Tensor.cpu(logits), axis=1)
        correct_predictions += sum(real == prediction)
        total_predictions += len(prediction)
    # print(visuals.numpy().max(), visuals.numpy().min())
    print("model accuracy on {} with normalization: {}".format(opt.dataset, correct_predictions/total_predictions))


if __name__ == '__main__':
    main()