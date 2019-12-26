import torch.nn as nn
import torchvision.models as models


class ClassificationCNN(nn.Module):
    def __init__(self):
        super(ClassificationCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)

        for param in resnet.parameters():
            param.requires_grad = False

        resnet.fc = nn.Linear(2048, 256)

        self.model = nn.Sequential(
            resnet,
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, images):
        return self.model(images).squeeze()
