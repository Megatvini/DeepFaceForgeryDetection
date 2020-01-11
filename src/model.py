import torchvision
from torch import nn
import resnet3d
from facenet_pytorch import InceptionResnetV1


class Lambda(nn.Module):
    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class CNN_LSTM(nn.Module):
    def __init__(self, image_encoding_size=128, hidden_size=128, pretrained_encoder=True):
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
        self.model = resnet3d.resnet10(num_classes=1)

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


class FaceRecognitionCNN(nn.Module):
    def __init__(self):
        super(FaceRecognitionCNN, self).__init__()
        self.resnet = InceptionResnetV1(pretrained='vggface2')
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 1)

    def forward(self, images):
        out = self.resnet(images)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out.squeeze()


class Encoder2DConv3D(nn.Module):
    def __init__(self):
        super(Encoder2DConv3D, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        self.encoder2d = nn.Sequential(*list(resnet.children())[:-3])
        self.encoder3d = nn.Sequential(
            nn.Conv3d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.Conv3d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

    def forward(self, images):
        batch_size, num_channels, depth, height, width = images.shape
        images = images.permute(0, 2, 1, 3, 4)
        images = images.reshape(batch_size * depth, num_channels, height, width)
        out = self.encoder2d(images)
        out = out.reshape(batch_size, depth, 256, 14, 14)
        out = out.permute(0, 2, 1, 3, 4)
        out = self.encoder3d(out)
        return out.squeeze()
