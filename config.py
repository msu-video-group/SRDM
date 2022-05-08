import os
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Path to dataset, could be overwritten
_C.DATA.DATA_PATH_TRAIN = '/mnt/hdd/datasets/SR-metric-dataset/VideoTrainDataset/'
_C.DATA.DATA_PATH_VAL = '/home/vyacheslav/mnt/mycalypso/Benchmark/ValidationDataset/Video/'
_C.DATA.SR_METHODS_TRAIN = ['Real-ESRGAN', 'RRN', 'RBPN', 'SOF_VSR', 'Topaz-4x', 'RealSRDataset', 'ESRGAN']
_C.DATA.SR_METHODS_VAL = ['Real-ESRGAN', 'RRN', 'RBPN', 'SOF_VSR', 'Topaz', 'LGFN']
# Input image size
_C.DATA.IMAGE_SIZE = 224
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Logger settings
# -----------------------------------------------------------------------------
_C.LOG = CN()
_C.LOG.DIR = 'exps'
_C.LOG.SAVED_MODELS = ''
_C.LOG.TB_LOGS = ''
_C.LOG.METRICS = ''

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name
_C.MODEL.NAME = 'resnet'
_C.MODEL.VERSION = ""
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 2
_C.MODEL.EMBEDDING_SIZE = 64
# Number of input consecutive frames
_C.MODEL.N_FRAMES = 2
# Loss
_C.MODEL.LOSS = CN()
_C.MODEL.LOSS.TRIPLET_MARGIN = 1.9
_C.MODEL.LOSS.CE = True
_C.MODEL.LOSS.STD = True
_C.MODEL.LOSS.TRP = True

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.NUM_EPOCHS = 200
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 2e-5
_C.TRAIN.WARMUP_LR = 5e-6
_C.TRAIN.MIN_LR = 5e-6
_C.TRAIN.ACCUMULATION_STEPS = 1
_C.TRAIN.PRINT_IMAGE_FREQ = 1000
_C.TRAIN.PRINT_TB_LOG_FREQ = 100

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.VAL = CN()
_C.VAL.BATCH_SIZE = 1
_C.VAL.FREQ = 10
_C.VAL.PRINT_IMAGE_FREQ = 10

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1
_C.TEST.DATA_PATH = ''
_C.TEST.QUANTILE = 0.1
_C.TEST.RESULT_PATH = 'results'

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Frequency to save checkpoint
_C.SAVE_FREQ = 10
# Fixed random seed
_C.SEED = 0


def update_config(config, args):
    config.defrost()

    config.MODEL.NAME = args.model_name
    config.MODEL.VERSION = args.version
    config.MODEL.LOSS.CE = args.ce
    config.MODEL.LOSS.TRP = args.trp
    config.MODEL.LOSS.STD = args.std
    config.MODEL.PRETRAINED = args.pretrained
    config.TRAIN.BATCH_SIZE = args.batch_size
    config.MODEL.EMBEDDING_SIZE = args.embedding_size
    config.MODEL.N_FRAMES = args.n_frames
    config.DATA.NUM_WORKERS = args.num_workers
    config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    config.TEST.DATA_PATH = args.test_data_path

    config.freeze()


def get_config(args):
    config = _C.clone()
    update_config(config, args)
    return config
