import torchvision
from torch import nn
from resnet3d import resnet10


class Lambda(nn.Module):
    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class CNN_LSTM(nn.Module):
    def __init__(self, image_encoding_size=512, hidden_size=512, pretrained_encoder=True):
        super(CNN_LSTM, self).__init__()
        self.image_encoding_size = image_encoding_size
        self.hidden_size = hidden_size
        self.cnn_encoder = nn.Sequential(
            torchvision.models.squeezenet1_1(pretrained=pretrained_encoder),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, image_encoding_size)
        )
        self.lstm = nn.LSTM(
            image_encoding_size, hidden_size, num_layers=1, bias=True, batch_first=True, bidirectional=True
        )
        self.relu = nn.ReLU()
        self.droupout = nn.Dropout(0.5)
        self.fc = nn.Linear(2*hidden_size, 1)

    def forward(self, images):
        batch_size, num_channels, depth, height, width = images.shape
        inp = images.permute(0, 2, 1, 3, 4).reshape(batch_size * depth, num_channels, height, width)
        image_encodings = self.cnn_encoder(inp).reshape(batch_size, depth, -1)
        out, _ = self.lstm(image_encodings)

        mid_frame = out[:, depth // 2, :]
        res = self.fc(self.droupout(self.relu(mid_frame)))
        return res.squeeze()


class ResNet3d(nn.Module):
    def __init__(self):
        super(ResNet3d, self).__init__()
        self.model = resnet10(num_classes=1)

    def forward(self, images):
        return self.model(images).squeeze()


class ResNet2d(nn.Module):
    def __init__(self, final_hidden_dim=256, dropout=0.5, pretrained=True):
        super(ResNet2d, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        resnet.fc = nn.Linear(512, final_hidden_dim)
        self.model = nn.Sequential(
            resnet,
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_hidden_dim, 1)
        )

    def forward(self, images):
        return self.model(images.squeeze()).squeeze()


class SqueezeNet2d(nn.Module):
    def __init__(self, dropout=0.5, pretrained=True):
        super(SqueezeNet2d, self).__init__()
        squeeze_net = torchvision.models.squeezenet1_1(pretrained=pretrained)
        self.model = nn.Sequential(
            squeeze_net,
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1000, 1),
        )

    def forward(self, images):
        return self.model(images.squeeze()).squeeze()


class Custom3DModel(nn.Module):
    def __init__(self):
        super(Custom3DModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(3, 16, 7, padding=3, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(64, 128, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(128, 1)
        )

    def forward(self, images):
        out = self.model(images)
        return out.squeeze()
