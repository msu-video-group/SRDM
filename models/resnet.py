import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models import resnet50


class SRDetectorResnet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, embedding_size=512):
        super().__init__()

        base_model = resnet50(pretrained=True)

        n_frames = n_channels // 3

        conv = base_model.conv1
        conv.in_channels = n_channels
        conv.weight = nn.Parameter(F.pad(conv.weight, [0, 0, 0, 0, 0, n_channels - 3], value=0),
                                   requires_grad=True)

        for i in range(1, n_frames):
            conv.weight[:, 3 * i: 3 * (i + 1)].data.copy_(conv.weight[:, :3].data)
        base_model.conv1 = conv

        features = []
        for module_pos, module in list(base_model._modules.items())[:-1]:
            features.append(module)

        self.feature_extractor = nn.Sequential(*features)

        self.projector = nn.Linear(in_features=2048, out_features=embedding_size, bias=True)

        self.projector = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=embedding_size, bias=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=embedding_size, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=n_classes, bias=True),
        )

        self.n_classes = n_classes
        self.embedding_size = embedding_size
        self.n_channels = n_channels

    def get_embeddings(self):
        return self.embeddings

    def forward(self, images: torch.Tensor):
        assert images.size()[-3] == self.n_channels

        X = images
        X = self.feature_extractor(X)
        X = torch.flatten(X, 1)
        X_embed = self.projector(X)
        self.embeddings = F.normalize(X_embed, dim=1)
        X_class = self.classifier(X_embed)
        return X_class
