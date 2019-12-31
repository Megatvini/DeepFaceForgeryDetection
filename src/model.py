import torch.nn as nn
import torchvision.models as models


class ClassificationCNN(nn.Module):
    def __init__(self):
        super(ClassificationCNN, self).__init__()
        resnet = models.resnet18(pretrained=False)

        final_hidden_dim = 256
        resnet.fc = nn.Linear(512, final_hidden_dim)

        self.model = nn.Sequential(
            resnet,
            nn.ReLU(),
            nn.Linear(final_hidden_dim, 1)
        )

    def forward(self, images):
        return self.model(images).squeeze()
