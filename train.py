import torch
import numpy as np
from tqdm import tqdm
from utils import visualize_images
import time

from metric import accuracy


def train(model, device, optimizer, criterion, scheduler, scaler, epoch, datasetTrain, writer_train, config):
    epoch_loss = []
    epoch_accuracy = []
    model.train()

    T = tqdm(enumerate(datasetTrain), desc=f'Epoch {epoch}')
    end = time.time()
    for i, (anchors, positives, negatives) in T:
        anchors = anchors.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)
        data_time = time.time() - end

        with torch.cuda.amp.autocast(enabled=True):
            input = torch.cat([anchors, positives, negatives])
            outputs = model(input)
            embeddings = model.get_embeddings()

            embeddings_anchors = embeddings[:anchors.shape[0]]
            embeddings_positives = embeddings[anchors.shape[0]:anchors.shape[0] + positives.shape[0]]
            embeddings_negatives = embeddings[anchors.shape[0] + positives.shape[0]:]

            labels = torch.tensor([0] * anchors.shape[0] +
                                  [0] * positives.shape[0] +
                                  [1] * negatives.shape[0], dtype=torch.long).to(device)

            loss = criterion(outputs, embeddings_anchors, embeddings_positives, embeddings_negatives, labels)
            scaler.scale(loss).backward()

        batch_time = time.time() - end
        end = time.time()

        batch_training_number = epoch * len(datasetTrain) + i
        if (batch_training_number + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()
            scheduler.step_update(batch_training_number)

        # statistics
        with torch.no_grad():
            epoch_loss.append(loss.cpu().detach().item() / config.TRAIN.ACCUMULATION_STEPS)
            epoch_accuracy.append(accuracy(outputs, labels).cpu().detach().item() / config.TRAIN.ACCUMULATION_STEPS)

            if batch_training_number % config.TRAIN.PRINT_IMAGE_FREQ == 0:
                writer_train.add_image('Images/train',
                                       visualize_images(torch.cat([positives, negatives]), labels[anchors.shape[0]:],
                                                        outputs[anchors.shape[0]:]),
                                       global_step=batch_training_number)
            T.set_description(f"Epoch {epoch}, loss: {np.mean(epoch_loss):.5f}, " + \
                              f"accuracy: {np.mean(epoch_accuracy):.5f}, data_time: {data_time:.3f}, " + \
                              f"batch_time: {batch_time:.3f}, scale: {scaler._scale.item()}, lr: {optimizer.param_groups[0]['lr']:.8f}",
                              refresh=False)
            if (batch_training_number + 1) % config.TRAIN.PRINT_TB_LOG_FREQ == 0:
                # Tensorboard batch
                writer_train.add_scalar('Loss/batch', np.mean(epoch_loss), global_step=batch_training_number)
                writer_train.add_scalar('Accuracy/batch', np.mean(epoch_accuracy), global_step=batch_training_number)
    writer_train.add_scalar('Loss/epoch', np.mean(epoch_loss), global_step=epoch)
    writer_train.add_scalar('Accuracy/epoch', np.mean(epoch_accuracy), global_step=epoch)
