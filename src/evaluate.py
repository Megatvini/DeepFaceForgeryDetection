import argparse

import numpy as np
import torch
import torch.nn as nn
from facenet_pytorch import fixed_image_standardization
from torchvision import transforms
from tqdm import tqdm

from data_loader import get_loader, read_dataset
from model import FaceRecognitionCNN


def read_testing_dataset(args, transform):
    datasets = read_dataset(
        args.data_dir, transform=transform,
        max_images_per_video=args.max_images_per_video, max_videos=args.max_videos,
        window_size=args.window_size, splits_path=args.splits_path
    )
    return datasets


def run_evaluate(args):
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406),
    #                          (0.229, 0.224, 0.225))
    # ])

    full_dataset = read_testing_dataset(args, transform)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('evaluating on', device)

    # Build the models
    model = FaceRecognitionCNN().to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    for test_dataset_name, dt in full_dataset.items():
        if 'c40' in test_dataset_name and ('original' in test_dataset_name or 'neural' in test_dataset_name):
            _, _, test_dataset = dt
            evaluate(args, device, model, test_dataset, test_dataset_name)


def evaluate(args, device, model, test_dataset, test_dataset_name):
    tqdm.write(f'evaluating for {test_dataset_name}')
    tqdm.write('test data size: {}'.format(len(test_dataset)))

    # Build data loader
    test_loader = get_loader(
        test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False
    )

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        loss_values = []
        all_predictions = []
        all_targets = []
        for video_ids, frame_ids, images, targets in tqdm(test_loader, desc=test_dataset_name):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            loss_values.append(loss.item())

            predictions = outputs > 0.0
            all_predictions.append(predictions)
            all_targets.append(targets)

        val_loss = sum(loss_values) / len(loss_values)

        all_predictions = torch.cat(all_predictions).int()
        all_targets = torch.cat(all_targets).int()
        test_accuracy = (all_predictions == all_targets).sum().float().item() / all_targets.shape[0]

        tqdm.write('Testing results - Loss: {:.3f}, Acc: {:.3f}'.format(val_loss, test_accuracy))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path for the model to evaluate')
    parser.add_argument('--data_dir', type=str, default='../dataset/images_tiny', help='directory for data with images')
    parser.add_argument('--max_images_per_video', type=int, default=999999, help='maximum images to use from one video')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--window_size', type=int, default=1)
    parser.add_argument('--max_videos', type=int, default=1000)
    parser.add_argument('--splits_path', type=str, default='../dataset/splits/')
    parser.add_argument('--comment', type=str, default='')
    args = parser.parse_args()
    run_evaluate(args)


if __name__ == '__main__':
    main()
