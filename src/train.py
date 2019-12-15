import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from data_loader import get_loader, read_dataset
from model import ClassificationCNN


def train(args):
    # show tensorboard graphs with following command: tensorboard --logdir =src/runs
    writer = SummaryWriter()

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    train_dataset, val_dataset = read_dataset(
        args.original_image_dir, args.tampered_image_dir, transform=transform,
        max_images_per_video=args.max_images_per_video
    )

    print('train data size: {}, validation data size: {}'.format(len(train_dataset), len(val_dataset)))

    # Build data loader
    train_loader = get_loader(
        train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = get_loader(
        val_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('training on', device)

    # Build the models
    model = ClassificationCNN().to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.regularization)

    training_images_sample, _ = next(iter(train_loader))
    grid = torchvision.utils.make_grid(training_images_sample)

    # Store training parameters
    writer.add_image('sample_training_images', grid, 0)
    writer.add_hparams(args.__dict__, {})
    writer.add_text('model', str(model))

    now = datetime.now()
    # Train the models
    total_step = len(train_loader)
    step = 1
    for epoch in range(args.num_epochs):
        for i, (images, targets) in enumerate(train_loader):
            model.train()
            # Set mini-batch dataset
            images = images.to(device)
            targets = targets.to(device)

            # Forward, backward and optimize
            outputs = model(images)
            loss = criterion(outputs, targets)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            batch_accuracy = float(outputs.round().eq(targets).sum()) / len(targets)

            iteration_time = datetime.now() - now
            # Print log info
            step += 1

            if (i + 1) % args.log_step == 0:
                print_training_info(args, batch_accuracy, epoch, i, iteration_time, loss, step, total_step, writer)

            if (i + 1) % args.val_step == 0:
                print_validation_info(args, criterion, device, model, val_loader, writer, step)
                save_model_checkpoint(args, epoch, i, model)

            now = datetime.now()

    print_validation_info(args, criterion, device, model, val_loader, writer, step + 1, final=True)
    writer.close()


def save_model_checkpoint(args, epoch, i, model):
    model_dir = args.model_path
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'{datetime.now()}_model_epoch_{epoch}_iter_{i}.pt')
    torch.save(model.state_dict(), model_path)
    print(f'New checkpoint saved at {model_path}')


def print_training_info(args, batch_accuracy, epoch, i, iteration_time, loss, step, total_step, writer):
    log_info = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}, Iteration time: {}'.format(
        epoch, args.num_epochs, i + 1, total_step, loss.item(), batch_accuracy, iteration_time
    )
    print(log_info)

    writer.add_scalar('training loss', loss.item(), step)
    writer.add_scalar('training acc', batch_accuracy, step)


def print_validation_info(args, criterion, device, model, val_loader, writer, step, final=False):
    now = datetime.now()
    model.eval()
    with torch.no_grad():
        loss_values = []
        correct_predictions = 0
        total_predictions = 0

        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            loss_values.append(loss.item())

            predictions = outputs.round()
            correct_predictions += int(targets.eq(predictions).sum().cpu())
            total_predictions += len(images)
            if args.debug:
                print(outputs)
                print(predictions)
                print(targets)

        val_loss = sum(loss_values) / len(loss_values)
        val_accuracy = correct_predictions / total_predictions
        print('Validation - Loss: {:.3f}, Acc: {:.3f}, Time: {}'.format(val_loss, val_accuracy, datetime.now() - now))
        writer.add_scalar('validation loss', val_loss, step)
        writer.add_scalar('validation acc', val_accuracy, step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument(
        '--original_image_dir', type=str, default='../dataset/images_tiny/original',
        help='directory for original images'
    )
    parser.add_argument(
        '--tampered_image_dir', type=str, default='../dataset/images_tiny/tampered',
        help='directory for tampered images'
    )
    parser.add_argument('--log_step', type=int, default=10, help='step size for printing training log info')
    parser.add_argument('--val_step', type=int, default=50, help='step size for printing validation log info')
    parser.add_argument('--max_images_per_video', type=int, default=10, help='maximum images to use from one video')
    parser.add_argument('--debug', type=bool, default=False, help='include additional debugging ifo')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')

    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--regularization', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=22)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.00001)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
