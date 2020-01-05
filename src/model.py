import torchvision
from torch import nn


class Lambda(nn.Module):
    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class ClassificationCNN(nn.Module):
    def __init__(self, image_encoding_size=512, hidden_size=512):
        super(ClassificationCNN, self).__init__()
        self.image_encoding_size = image_encoding_size
        self.hidden_size = hidden_size
        self.cnn_encoder = nn.Sequential(
            torchvision.models.squeezenet1_1(pretrained=True),
            nn.ReLU(),
            nn.Linear(1000, image_encoding_size)
        )
        self.lstm = nn.LSTM(
            image_encoding_size, hidden_size, num_layers=1, bias=True, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(2*hidden_size, 1)

    def forward(self, images):
        batch_size, num_channels, depth, height, width = images.shape
        inp = images.permute(0, 2, 1, 3, 4).reshape(batch_size * depth, num_channels, height, width)
        image_encodings = self.cnn_encoder(inp).reshape(batch_size, depth, -1)
        out, _ = self.lstm(image_encodings)

        mid_frame = out[:, depth // 2, :]
        res = self.fc(mid_frame)
        return res.squeeze()
