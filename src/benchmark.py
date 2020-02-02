import argparse
import os

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import fixed_image_standardization
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from model import FaceRecognitionCNN
from utils import write_json
from facenet_pytorch import MTCNN

FACES_DATA_DIR = '../dataset/faceforensics_benchmark_faces'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on', device)


class ImagesDataset(Dataset):
    def __init__(self, images_dir, transform) -> None:
        super().__init__()
        self._images = []
        self.read_images(images_dir)
        self.transform = transform

    def __getitem__(self, index: int):
        image_name, image_path = self._images[index]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image_name, image

    def __len__(self) -> int:
        return len(self._images)

    def read_images(self, images_dir):
        for image_name in os.listdir(images_dir):
            image_path = os.path.join(images_dir, image_name)
            self._images.append((image_name, image_path))


def run_evaluate(model_path):
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    full_dataset = ImagesDataset(FACES_DATA_DIR, transform)

    # Build the models
    model = FaceRecognitionCNN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    res = {}
    with torch.no_grad():
        for image_name, image in tqdm(full_dataset, desc='Evaluating frames'):
            image = image.to(device)
            output = model(image.unsqueeze(0)).item()
            prediction = 'fake' if output > 0.0 else 'real'
            res[image_name] = prediction

    write_json(res, 'benchmark.json')


def extract_faces(data_dir):
    face_detector = MTCNN(device=device, margin=16)
    face_detector.eval()
    for image_name in tqdm(os.listdir(data_dir), desc='Extracting faces'):
        inp_img_path = os.path.join(data_dir, image_name)
        out_img_path = os.path.join(FACES_DATA_DIR, image_name)
        if not os.path.exists(out_img_path):
            image = Image.open(inp_img_path)
            face_detector(image, save_path=out_img_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path for the model to evaluate')
    parser.add_argument('data_dir', type=str, help='path to images to classify')
    args = parser.parse_args()
    extract_faces(args.data_dir)
    run_evaluate(args.model_path)


if __name__ == '__main__':
    main()
