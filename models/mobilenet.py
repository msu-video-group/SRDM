import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


class SRDetectorMobilenet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, embedding_size=512):
        super().__init__()

        base_model = mobilenet_v2(pretrained=True)

        if n_channels == 6:
            conv = base_model.features[0][0]
            conv.in_channels = n_channels
            conv.weight = nn.Parameter(F.pad(conv.weight, [0, 0, 0, 0, 0, conv.in_channels - 3], value=0),
                                       requires_grad=True)
            conv.weight[:, 3:].data.copy_(conv.weight[:, :3].data)
            base_model.conv1 = conv

        features = []
        for module_pos, module in list(base_model.features._modules.items()):
            features.append(module)

        self.feature_extractor = nn.Sequential(*features)

        self.projector = nn.Linear(in_features=1280, out_features=embedding_size, bias=True)

        self.projector = nn.Sequential(
            nn.Linear(in_features=1280, out_features=1024, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=embedding_size, bias=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=1280, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=n_classes, bias=True),
        )

        self.n_classes = n_classes
        self.embedding_size = embedding_size
        self.n_channels = n_channels

    def finetune_step1(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        for param in self.projector.parameters():
            param.requires_grad = True

        for param in self.classifier.parameters():
            param.requires_grad = True

    def finetune_step2(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        for param in self.projector.parameters():
            param.requires_grad = True

        for param in self.classifier.parameters():
            param.requires_grad = True

    def get_embeddings(self):
        return self.embeddings

    def forward(self, images: torch.Tensor):
        assert images.size()[-3] == self.n_channels

        X = images
        X = self.feature_extractor(X)
        X = nn.functional.adaptive_avg_pool2d(X, (1, 1))
        X = torch.flatten(X, 1)
        X_embed = self.projector(X)
        X_embed = F.normalize(X_embed, dim=1)
        self.embeddings = X_embed
        X_class = self.classifier(X)
        return X_class
