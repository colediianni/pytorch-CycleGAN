import os
import torch
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
# from util.visualizer import save_images
# from util import html
import matplotlib.pyplot as plt

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
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results. Stored as real_X, fake_Y, rec_X (reconstructed)
        img_path = model.get_image_paths()     # get image paths
        
        images = ["real_A", "fake_B", "rec_A", "real_B", "fake_A", "rec_B"]
        for img_index in range(6):
            plt.subplot(2, 3, img_index+1)
            plt.title(images[img_index])
            image = np.transpose(visuals[images[img_index]][0].cpu().detach().numpy(), (1, 2, 0))
            image = (image / 2) + 0.5 # Unnormalize because current image is between -1 and 1
            plt.imshow(image)
        plt.savefig(os.path.join(opt.results_dir, img_path[0][img_path[0].rfind('/') + 1:]))
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        
            
        # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)