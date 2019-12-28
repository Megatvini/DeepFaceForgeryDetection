import torch.nn as nn
import torchvision.models as models


class ClassificationCNN(nn.Module):
    def __init__(self):
        super(ClassificationCNN, self).__init__()
        resnet = models.resnet18(pretrained=True)

        final_hidden_dim = 256
        resnet.fc = nn.Linear(512, final_hidden_dim)

        self.model = nn.Sequential(
            resnet,
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(final_hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, images):
        return self.model(images).squeeze()
