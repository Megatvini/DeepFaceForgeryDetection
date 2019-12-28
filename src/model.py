import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class ClassificationCNN(nn.Module):
    def __init__(self):
        super(ClassificationCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(3, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(224*224*5, 1),
            nn.Sigmoid()
        )

    def forward(self, images):
        out = self.model(images)
        return out.squeeze()
