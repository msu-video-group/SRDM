import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, config=None):
        super(TripletLoss_std, self).__init__()
        self.ce_flag = config.MODEL.LOSS.CE
        self.trp_flag = config.MODEL.LOSS.TRP
        self.std_flag = config.MODEL.LOSS.STD
        self.margin = config.MODEL.LOSS.TRIPLET_MARGIN

    def triplet_loss(self, anchors, positives, negatives):
        triplet_loss_func = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=self.margin)
        loss = triplet_loss_func(anchors, positives, negatives)
        return loss

    def crossentropy_loss(self, outputs, anchors, positives, negatives, labels):
        clf_loss = 1 / 4 * F.cross_entropy(outputs[:anchors.shape[0]], labels[:anchors.shape[0]]) + \
                   1 / 4 * F.cross_entropy(outputs[anchors.shape[0]: anchors.shape[0] + positives.shape[0]],
                                           labels[anchors.shape[0]: anchors.shape[0] + positives.shape[0]]) + \
                   1 / 2 * F.cross_entropy(outputs[anchors.shape[0] + positives.shape[0]:],
                                           labels[anchors.shape[0] + positives.shape[0]:])
        return clf_loss

    def forward(self, outputs, anchors, positives, negatives, labels):
        assert anchors.shape[0] == positives.shape[0] == negatives.shape[0]
        assert outputs.shape[0] == labels.shape[0]

        loss = 0
        if self.ce_flag:
            loss += self.crossentropy_loss(outputs, anchors, positives, negatives, labels)
        if self.trp_flag:
            loss += self.triplet_loss(anchors, positives, negatives)
        return loss
