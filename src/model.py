from resnet3d import resnet10
from torch import nn


class ClassificationCNN(nn.Module):
    def __init__(self):
        super(ClassificationCNN, self).__init__()
        self.model = resnet10(num_classes=1)

    def forward(self, images):
        out = self.model(images)
        return out.squeeze()
