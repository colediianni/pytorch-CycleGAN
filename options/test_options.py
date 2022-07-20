from .base_options import BaseOptions

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

class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=20, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        
        parser.add_argument('--dataset', '-d', choices=DATASETS,
                            default="Test_GAN_Normalized", help='test set to evaluate model on')
        parser.add_argument('--model-type', '-mt', choices=['unnormalized', 'gan-normalized'], default='unnormalized', help='model to evaluate')
        parser.add_argument('--batch-size', '-b', type=int,
                            default=BATCH_SIZE_DEF, help='batch size for the data loader')
        parser.add_argument('--no-cuda', action='store_true',
                            default=False, help='disables CUDA training')
        parser.add_argument('--seed', '-s', type=int, default=SEED_DEFAULT,
                            help='random seed (default: 123)')
        parser.add_argument('--n-jobs', type=int, default=32,
                            help='number of parallel jobs to use for multiprocessing')
        parser.add_argument('--gpu', type=str, default='1',
                            help='gpus to execute code on')
        parser.add_argument('--output-dir', '-o', type=str,
                            default=OUTPUT_PATH, help='output directory path')
        
        self.isTrain = False
        return parser
