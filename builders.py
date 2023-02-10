import os
import torch
import datetime
from timm.scheduler.cosine_lr import CosineLRScheduler

from models.resnet import SRDetectorResnet
from models.mobilenet import SRDetectorMobilenet
from loss import TripletLoss


def build_logger(config):
    logger_dir = config.LOG.DIR
    saved_models_dir = f'{logger_dir}/saved_models/{config.MODEL.NAME}-{config.MODEL.VERSION}_{datetime.datetime.now().strftime("%Y%m%d")}/'
    tb_writer_dir = f'{logger_dir}/tensorboard_logs/{config.MODEL.NAME}-{config.MODEL.VERSION}_{datetime.datetime.now().strftime("%Y%m%d")}/'
    metrics_dir = f'{logger_dir}/counted_metrics/{config.MODEL.NAME}-{config.MODEL.VERSION}_{datetime.datetime.now().strftime("%Y%m%d")}/'

    os.makedirs(saved_models_dir, exist_ok=True)
    os.makedirs(tb_writer_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    config.defrost()
    config.LOG.SAVED_MODELS = saved_models_dir
    config.LOG.TB_LOGS = tb_writer_dir
    config.LOG.METRICS = metrics_dir
    config.freeze()


def build_model(config):
    model_name = config.MODEL.NAME
    if model_name.startswith("resnet"):
        model = SRDetectorResnet(n_classes=config.MODEL.NUM_CLASSES, n_channels=int(config.MODEL.N_FRAMES * 3),
                                 embedding_size=config.MODEL.EMBEDDING_SIZE)
    elif model_name.startswith("mobilenet"):
        model = SRDetectorMobilenet(n_classes=config.MODEL.NUM_CLASSES, n_channels=int(config.MODEL.N_FRAMES * 3),
                                    embedding_size=config.MODEL.EMBEDDING_SIZE)
    else:
        raise Exception("Unknown model name")
    if config.MODEL.PRETRAINED:
        model.load_state_dict(torch.load(config.MODEL.PRETRAINED)['model'])
    return model


def build_optimizer(parameters, config):
    optimizer = torch.optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                  lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    if config.MODEL.PRETRAINED:
        optimizer.load_state_dict(torch.load(config.MODEL.PRETRAINED)['optimizer'])
    return optimizer


def build_scheduler(optimizer, config, n_iter_per_epoch):
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=int(config.TRAIN.NUM_EPOCHS * n_iter_per_epoch),
        lr_min=config.TRAIN.MIN_LR,
        warmup_lr_init=config.TRAIN.WARMUP_LR,
        warmup_t=int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch),
        cycle_limit=1,
        t_in_epochs=False,
    )
    if config.MODEL.PRETRAINED:
        scheduler.load_state_dict(torch.load(config.MODEL.PRETRAINED)['scheduler'])
    return scheduler


def build_scaler(config):
    scaler = torch.cuda.amp.GradScaler()
    if config.MODEL.PRETRAINED:
        scaler.load_state_dict(torch.load(config.MODEL.PRETRAINED)['scaler'])
    return scaler


def build_epoch(config):
    epoch = config.TRAIN.START_EPOCH
    if config.MODEL.PRETRAINED:
        epoch = torch.load(config.MODEL.PRETRAINED)['epoch']
    return epoch


def build_criterion(config):
    return TripletLoss(config)
