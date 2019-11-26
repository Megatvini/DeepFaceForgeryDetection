import torch.nn as nn
import torchvision.models as models


class ClassificationCNN(nn.Module):
    def __init__(self):
        super(ClassificationCNN, self).__init__()
        resnet = models.resnet50(pretrained=False)
        resnet.fc = nn.Linear(2048, 1)
        self.model = nn.Sequential(
            resnet,
            nn.Sigmoid()
        )

    def forward(self, images):
        return self.model(images).squeeze()
