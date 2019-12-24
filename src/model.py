import torch.nn as nn
import torchvision.models as models


class ClassificationCNN(nn.Module):
    def __init__(self):
        super(ClassificationCNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 3, kernel_size=3)
        self.conv2 = nn.Conv3d(3, 1, kernel_size=3)
        self.fc = nn.Linear(2048, 1)

    def forward(self, images):
        out = self.conv1(images)
        out = self.conv2(out)
        out = out.view(-1)
        out = self.fc(out)
        return out
