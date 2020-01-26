import torch
import torchvision
from facenet_pytorch import InceptionResnetV1
from torch import nn

import resnet3d


class NNLambda(nn.Module):
    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class CNN_LSTM(nn.Module):
    def __init__(self, face_recognition_cnn_path, hidden_size=64):
        super(CNN_LSTM, self).__init__()
        image_encoding_size = 64

        face_cnn = FaceRecognitionCNN()
        state_dict = torch.load(face_recognition_cnn_path, map_location='cpu')
        face_cnn.load_state_dict(state_dict)
        face_cnn = nn.Sequential(*list(face_cnn.resnet.children()))[:-12]

        for p in face_cnn.parameters():
            p.requires_grad_(False)

        self.cnn_encoder = nn.Sequential(
            face_cnn,

            nn.Conv2d(192, 128, 5, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, image_encoding_size, 5, bias=False),
            nn.BatchNorm2d(image_encoding_size),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.lstm = nn.LSTM(
            image_encoding_size, hidden_size, num_layers=2, bias=True,
            batch_first=True, bidirectional=True, dropout=0.5
        )
        self.fc = nn.Linear(2*hidden_size, 1)

    def forward(self, images):
        batch_size, num_channels, depth, height, width = images.shape
        inp = images.permute(0, 2, 1, 3, 4).reshape(batch_size * depth, num_channels, height, width)

        inp = self.cnn_encoder(inp).reshape(batch_size, depth, -1)
        inp = self.relu1(inp)
        inp = self.dropout1(inp)

        out, _ = self.lstm(inp)

        mid_out = out[:, depth // 2, :]

        res = self.fc(mid_out)
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
    def __init__(self, face_recognition_cnn_path=None):
        super(Encoder2DConv3D, self).__init__()

        face_cnn = FaceRecognitionCNN()
        if face_recognition_cnn_path is not None:
            state_dict = torch.load(face_recognition_cnn_path, map_location='cpu')
            face_cnn.load_state_dict(state_dict)

        self.encoder2d = nn.Sequential(*list(face_cnn.resnet.children()))[:-10]
        self.encoder3d = nn.Sequential(
            nn.Conv3d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.Conv3d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, images):
        batch_size, num_channels, depth, height, width = images.shape
        images = images.permute(0, 2, 1, 3, 4)
        images = images.reshape(batch_size * depth, num_channels, height, width)
        out = self.encoder2d(images)
        out = out.reshape(batch_size, depth, 256, 17, 17)
        out = out.permute(0, 2, 1, 3, 4)
        out = self.encoder3d(out)
        return out.squeeze()


class MajorityVoteModel(nn.Module):
    def __init__(self, face_recognition_cnn_path):
        super(MajorityVoteModel, self).__init__()

        face_cnn = FaceRecognitionCNN()
        state_dict = torch.load(face_recognition_cnn_path, map_location='cpu')
        face_cnn.load_state_dict(state_dict)

        self.cnn_encoder = face_cnn

    def forward(self, images):
        batch_size, num_channels, depth, height, width = images.shape
        images = images.permute(0, 2, 1, 3, 4)
        images = images.reshape(batch_size * depth, num_channels, height, width)
        out = self.cnn_encoder(images)
        out = out.reshape(batch_size, depth)
        out = ((out > 0.0).sum(axis=1) > depth // 2).float()
        # out = out[:, depth//2]
        return out
