import torch.nn as nn
import torchvision.models as models


class ClassificationCNN(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ClassificationCNN, self).__init__()
        self.model = models.resnet18(pretrained=True)

    def forward(self, images):
        return self.model(images)
