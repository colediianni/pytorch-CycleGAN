import numpy as np
import pandas as pd
import io
import time
import cv2
import pathlib
import random
from PIL import Image
from matplotlib import pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool
from numpy.random import RandomState
import sklearn
from tqdm import tqdm
from collections import defaultdict
import albumentations as A
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms, utils
from skimage import io, transform
import matplotlib.pyplot as plt
import argparse
from skimage import data
from skimage.color import rgb2hed, hed2rgb
from matplotlib.colors import LinearSegmentedColormap
import pretrainedmodels
import base64
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torchvision.models as models
from collections import defaultdict
from collections import Counter
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import copy
from datasets.glioma_data_loader import WSIDataset
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


class My_Transform(object):
    '''
    My transform: 
    '''

    def __init__(self):
        pass

    def __call__(self, sample):
        image, label, p_id, path = sample['data'], sample['target'], sample['p_id'], sample['path']
        aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])

        augmented = aug(image=image)
        image_medium = augmented['image']
        # print('augment ',image_medium.shape)

        return {'data': image_medium, 'target': label, 'p_id': p_id, 'path': path}
    
class My_Transform_valid(object):
    '''
    My transform: 
    '''
    def __init__(self):
        pass

    def __call__(self, sample):
        image, label, p_id, path = sample['data'], sample['target'], sample['p_id'], sample['path']
        aug = A.Compose([
            A.Resize(256, 256, p=1)    
         ])

        augmented = aug(image=image)
        image_medium = augmented['image']
        return {'data': image_medium, 'target': label, 'p_id': p_id, 'path': path}

class My_Normalize(object):
    '''
    My Normalize (TRail)
    '''

    def __init__(self):
        pass

    def __call__(self, sample):
        image, label, p_id, path = sample['data'], sample['target'], sample['p_id'], sample['path']
        normal_aug = A.Normalize()
        augmented_img = normal_aug(image=image)
        image = augmented_img['image']
        image = image.transpose((2, 0, 1))
        # image = image/255.0
        # print('normal ',image.shape)
        return {'data': torch.from_numpy(image), 'target': torch.FloatTensor([label]), 'p_id': p_id, 'path': path}
    
class Cyclegan_Normalize(object):
    '''
    My Normalize (TRail)
    '''
    def __init__(self):
        pass

    def __call__(self, sample):
        image, label, p_id, path = sample['data'], sample['target'], sample['p_id'], sample['path']
        # print(image)
        # print(image.shape)
        normal_aug = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        augmented_img = normal_aug(image)
        # print(augmented_img)
        # print(augmented_img.shape)
        image = augmented_img
        # image = image.transpose((2, 0, 1))
        # image = image/255.0
        # print('normal ',image.shape)
        # return {'image': torch.from_numpy(image), 'label':torch.FloatTensor([label]), 'pid':p_id}
        return {'data': image, 'target': label, 'p_id': p_id, 'path': path}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # print(list(sample.keys()))
        image, label, p_id, path = sample['data'], sample['target'], sample['p_id'], sample['path']
        image = image.transpose((2, 0, 1))
        return {'data': torch.from_numpy(image).float(),
                'target':label, # .squeeze(0),
                'p_id': p_id,
                'path': path}

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
# class Identity(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#     def forward(self, x):
#         return x
    
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet152
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
        # model_ft = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        # print(num_ftrs, num_classes)
        model_ft.fc = nn.Linear(num_ftrs,num_classes) 
        input_size = 1024
    elif model_name == "inceptresnet":
        """ inceptresnetv2
        """
        model_ft = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(13824, num_classes)
        input_size = 1024 
    elif model_name == "senet":
        """ inceptresnetv2
        """
        model_ft = pretrainedmodels.__dict__['senet154'](num_classes=1000, pretrained='imagenet')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(1384448, num_classes)
        input_size = 1024 
    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 1024

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 1024

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 1024

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 1024

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    params_to_update = model_ft.parameters()
    return model_ft, input_size

def test_model(t_model, testloader, device, weight_trained=None):
    since = time.time()
    if weight_trained:
        t_model.load_state_dict(torch.load(weight_trained))
    t_model.eval()
    p_classification = defaultdict(list)
    p_classification_prob = defaultdict(list)
    p_label = {}
    TP = TN = FP = FN = 1e-4
    slide_label = []
    slide_prob = []
    for data in tqdm(testloader):
        inputs = data['data'].to(device=device, dtype=torch.float)
        labels = torch.unsqueeze(data['target'].to(device=device, dtype=torch.float32), dim=1)
        labels = torch.cat([1 - labels, labels], dim=1)
        _, label_index = torch.max(labels, 1)
        p_id = data['p_id']
        output = t_model(inputs)
        _, prediction = torch.max(output, 1)
        for i in range(output.size()[0]):
            curr_p_id = p_id[i]#.item()
            p_label[curr_p_id] = label_index[i].cpu().item()
            p_classification[curr_p_id].append(prediction[i].cpu().item())
            p_classification_prob[curr_p_id].append([output[i][0].cpu().item(), output[i][1].cpu().item()])
            if prediction[i].cpu().item() == label_index[i].cpu().item() == 0:
                TN += 1
            elif prediction[i].cpu().item() == label_index[i].cpu().item()  == 1:
                TP += 1
            elif prediction[i].cpu().item() == 0 and label_index[i].cpu().item()  == 1:
                FN += 1
            else:
                FP += 1
            slide_label.append(label_index[i].cpu().item())
            slide_prob.append(output[i][1].cpu().item())
    Specificity = TN / (TN + FP)#Specificity = TN / (TN + FP)
    Sensitivity = TP / (FN + TP)#Sensitivity = TP / (FN + TP)
    Precision = TP / (TP + FP)#Precision = TP / (TP + FP)
    F1_Score = 2*(Precision * Sensitivity) / (Precision + Sensitivity)
    AUC = roc_auc_score(slide_label, slide_prob)
    print(TP, TN, FN, FP)
    print('Patch level Statistic: ')
    print('Specificity: ', Specificity)
    print('Sensitivity: ', Sensitivity)
    print('Precision: ', Precision)
    print('Acc: ', (TP+TN)/(TP+TN+FN+FP))
    print('F1-Score: ',F1_Score)
    print('AUC ',AUC)
    return p_label, p_classification, p_classification_prob

def patent_class(p_label, p_classification, p_classification_prob, FP_lst, FN_lst):
    p_classification_check = defaultdict(list)

    TP = TN = FP = FN = 1e-4
    pat_label = []
    pat_pred = []
    pat_prob = []
    for k in p_classification:
        #classification of slide for one patient
        p_list = p_classification[k]
        x = Counter(p_list)
        s = sum(p_list)
        #majority vote for patient level classification
        p_pred = x.most_common(1)[0][0]
        #keep the prob for the positive label
        p_prob = p_classification_prob[k]
        p_pos = list(map(lambda x:x[1], p_prob))
        p_pos = sum(p_pos)/len(p_pos)
        pat_prob.append(p_pos)
        p_gt = p_label[k]
        if p_pred == p_gt == 0:
            TN += 1
        elif p_pred == p_gt  == 1:
            TP += 1
        elif p_pred == 0 and p_gt  == 1:
            FN += 1
            if k not in FN_lst:
                FN_lst[k] = 1
            else:
                FN_lst[k] += 1
        else:
            FP += 1
            if k not in FP_lst:
                FP_lst[k] = 1
            else:
                FP_lst[k] += 1
        pat_label.append(p_gt)
        pat_pred.append(p_pred)
    Specificity = TN / (TN + FP)#Specificity = TN / (TN + FP)
    Sensitivity = TP / (FN + TP)#Sensitivity = TP / (FN + TP)
    Precision = TP / (TP + FP)#Precision = TP / (TP + FP)
    F1_Score = 2*(Precision * Sensitivity) / (Precision + Sensitivity)
    # print(pat_label)
    # print(pat_prob)
    AUC = roc_auc_score(pat_label, pat_prob)
    ns_fpr, ns_tpr, _ = roc_curve(pat_label, pat_prob)
    print(TP, TN, FN, FP)
    print('FN ',FN_lst)
    print('FP ',FP_lst)
    print('Statistic: ')
    print('Specificity: ', Specificity)
    print('Sensitivity: ', Sensitivity)
    print('Precision: ', Precision)
    print('Acc: ', (TP+TN)/(TP+TN+FN+FP))
    print('F1-Score: ',F1_Score)
    print('AUC ',AUC)
    return (TP+TN)/(TP+TN+FN+FP), F1_Score, AUC, FP_lst, FN_lst

def train_model(model, trainloader, valloader, model_save_pth, model_name, device, num_epochs=40, is_inception=False):
    since = time.time()
    model = model.to(device)
    dataloaders = {'train': trainloader, 'val1': valloader}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()#nn.CrossEntropyLoss()#
    val_acc_history = []
    train_acc_history = []
    FP_lst = {} 
    FN_lst = {}
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val1']:
        # for phase in ['val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for data in tqdm(dataloaders[phase]):
                    inputs = data['data'].to(device, dtype=torch.float)
                    labels = torch.unsqueeze(data['target'].to(device=device, dtype=torch.float32), dim=1)
                    # print(labels)
                    # print(labels.shape)
                    labels = torch.cat([1 - labels, labels], dim=1)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        if is_inception and phase == 'train':
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        

                    # statistics
                    _, label_index = torch.max(labels, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == label_index.data)
                    # break
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                scheduler.step(epoch_loss)
                
            else:
                model.eval()   # Set model to evaluate mode
                p_label, p_classification,p_classification_prob = test_model(model, valloader, device)
                acc, f1, auc, FP_lst, FN_lst = patent_class(p_label, p_classification, p_classification_prob, FP_lst, FN_lst)
                print('Current ACC ',acc)
                print('Best ACC ',best_acc)
                # deep copy the model
                if acc > best_acc:
                    best_acc = acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, model_save_pth+model_name+'.pt')
                        
            if phase == 'val1':
                val_acc_history.append(epoch_acc)
            if phase == 'train':
                train_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, train_acc_history, val_acc_history

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-dataset', default="Train_Unnormalized", help='dataset to use for training')
    parser.add_argument('--validation-dataset', default="Valid_Unnormalized", help='dataset to use for validation')
    parser.add_argument('--testing-dataset', default="Test_Unnormalized", help='dataset to use for testing')
    parser.add_argument('--model-name', help='name of model')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size of evaluation')
    args = parser.parse_args()
    
    # TODO: Update these
    # train_path = '/content/drive/MyDrive/TCGA_926_Converted_Patched_Filtered/_Train_All.csv'
    # test_path = '/content/drive/MyDrive/TCGA_926_Converted_Patched_Filtered/_Test_All.csv'

    # batch_size = 16
    # validation_split = .3
    # shuffle_dataset = True
    # random_seed= 42
    # train_df = pd.read_csv(train_path)
    # valid_df = pd.read_csv(test_path)
    

    train_wsi_dataset = WSIDataset(GLIOMA_MODEL_PATH, transform=transforms.Compose([My_Transform(),
                                                                                    ToTensor(),
                                                                                    Cyclegan_Normalize()]))
    val_wsi_dataset = WSIDataset(GLIOMA_MODEL_PATH, transform=transforms.Compose([My_Transform(),
                                                                                  ToTensor(),
                                                                                  Cyclegan_Normalize()]))
    test_wsi_dataset = WSIDataset(GLIOMA_MODEL_PATH,transform=transforms.Compose([My_Transform_valid(),
                                                                                  ToTensor(),
                                                                                  Cyclegan_Normalize()]))
    
    trainloader = train_wsi_dataset.Obtain_loader(args.training_dataset, batch_size=args.batch_size, n_jobs=4)
    valloader = val_wsi_dataset.Obtain_loader(args.validation_dataset, batch_size=args.batch_size, n_jobs=4)
    testloader = test_wsi_dataset.Obtain_loader(args.testing_dataset, batch_size=args.batch_size, n_jobs=4)

    print('train {} vs valid {} vs test {}'.format(len(trainloader), len(valloader), len(testloader)))

    # Initialize the model for this run
    model_ft, input_size = initialize_model('resnet', 2, False, use_pretrained=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('\n Using : ', device)
    # model_save_pth = '/content/drive/MyDrive/stain_norm/results/unet_resunet_breakhis/normalised/%d/fold%d'%(sn_epoch, fold)
    model_save_pth = './models/'

    model_trained, train_acc_history, val_acc_history = train_model(model_ft, trainloader, valloader, model_save_pth, args.model_name, device, num_epochs=40)
    
    test_model(model_trained, testloader, device, weight_trained=None)


if __name__ == '__main__':
    main()