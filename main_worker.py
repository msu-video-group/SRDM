import os
import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

from train import train
from validation import validate
from dataloader import build_dataloader
from builders import build_logger, build_model, build_optimizer, build_scheduler, build_scaler, build_epoch, \
    build_criterion
from utils import save_model, parse_option


def main_worker(config):
    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    build_logger(config)
    datasetTrain, datasetValidVideoOriginal, datasetValidVideoCompressed = build_dataloader(config)
    writer_train = SummaryWriter(os.path.join(config.LOG.TB_LOGS, 'train'))
    writer_valid = SummaryWriter(os.path.join(config.LOG.TB_LOGS, 'valid'))

    model = build_model(config).to(device)
    optimizer = build_optimizer(model.parameters(), config)
    scheduler = build_scheduler(optimizer, config, len(datasetTrain))
    scaler = build_scaler(config)
    epoch = build_epoch(config)
    criterion = build_criterion(config)

    for epoch in range(epoch, config.TRAIN.NUM_EPOCHS):
        train(model, device, optimizer, criterion, scheduler, scaler, epoch, datasetTrain, writer_train, config)
        if (epoch + 1) % config.VAL.FREQ == 0:
            validate(model, datasetValidVideoOriginal, datasetValidVideoCompressed, device, writer_valid, epoch, config)
        if (epoch + 1) % config.SAVE_FREQ == 0:
            save_model(model, optimizer, scheduler, scaler, epoch, config)


if __name__ == "__main__":
    config = parse_option()
    print("Config:"
          "\n-------------------------------------------------------------\n",
          config,
          "\n-------------------------------------------------------------\n", sep="")
    main_worker(config)
