import torchvision
from torch import nn
import resnet3d
from facenet_pytorch import InceptionResnetV1
import torch


class CNN_LSTM(nn.Module):
    def __init__(self, face_recognition_cnn_path, hidden_size=512):
        super(CNN_LSTM, self).__init__()
        image_encoding_size = 512

        face_cnn = FaceRecognitionCNN()
        state_dict = torch.load(face_recognition_cnn_path, map_location='cpu')
        face_cnn.load_state_dict(state_dict)

        self.cnn_encoder = face_cnn.resnet
        self.lstm = nn.LSTM(
            image_encoding_size, hidden_size, num_layers=2, bias=True, batch_first=True, bidirectional=True
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2*hidden_size, 1)

    def forward(self, images):
        batch_size, num_channels, depth, height, width = images.shape
        inp = images.permute(0, 2, 1, 3, 4).reshape(batch_size * depth, num_channels, height, width)
        image_encodings = self.cnn_encoder(inp).reshape(batch_size, depth, -1)
        out, _ = self.lstm(image_encodings)

        mid_out = out[:, depth // 2, :]
        res = self.fc(self.dropout(self.relu(mid_out)))
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
    def __init__(self, face_recognition_cnn_path):
        super(Encoder2DConv3D, self).__init__()

        face_cnn = FaceRecognitionCNN()
        state_dict = torch.load(face_recognition_cnn_path, map_location='cpu')
        face_cnn.load_state_dict(state_dict)

        self.cnn_encoder = nn.Sequential(*list(face_cnn.resnet.children()))[:-10]
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
        out = self.cnn_encoder(images)
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
