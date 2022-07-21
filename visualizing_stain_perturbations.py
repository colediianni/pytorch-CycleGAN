import os
import torch
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
# from util.visualizer import save_images
# from util import html
import matplotlib.pyplot as plt
from tqdm import tqdm
import histomicstk
import cv2

from helpers.utils import (
    load_model_checkpoint,
    get_num_jobs
    #     extract_layer_embeddings,
    #     get_activations
)

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

from datasets.glioma_data_loader import (
    #     test_model,
    WSIDataset,
    My_Normalize
    #     patent_class
)

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    
    use_cuda = not opt.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    wsi_dataset = WSIDataset(GLIOMA_MODEL_PATH)

    n_jobs = get_num_jobs(opt.n_jobs)
    test_data_loader = wsi_dataset.Obtain_loader(
        opt.dataset, opt.batch_size, n_jobs)
    
    i = 0
    for image_data in tqdm(test_data_loader):
        img_path = image_data['path'][0]
        # visuals = image_data['data'].to(device=device, dtype=torch.float)
        visuals = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        visuals= cv2.resize(visuals, (256, 256), interpolation=cv2.INTER_CUBIC)
        if i >= 10:  # only perturb 10 images
            break
        i += 1
        
        perturbations = 9
        for img_index in range(perturbations):
            plt.subplot(3, 3, img_index+1)
            if img_index == 0:
                plt.title("original")
            else:
                plt.title("perturbation " + str(img_index))
            image = visuals
            # print(image)
            # print(image.shape)
            # image = np.transpose(image, (1, 2, 0))
            # print(image)
            if img_index != 0:
                image = histomicstk.preprocessing.augmentation.rgb_perturb_stain_concentration(image)
                print(image)
            plt.imshow(image)
        print(img_path)
        plt.savefig(os.path.join(opt.results_dir, "stain_perturbed_" + img_path[img_path.rfind('/') + 1:]))
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        
            
        # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)