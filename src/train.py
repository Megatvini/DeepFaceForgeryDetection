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
from utils import write_json, copy_file


def train(args):
    # show tensorboard graphs with following command: tensorboard --logdir =src/runs
    writer = SummaryWriter()

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
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
        val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('training on', device)

    # Build the models
    model = ClassificationCNN().to(device)

    print(f"Total parameters of model: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.regularization
    )

    # decrease learning rate if validation accuracy has not increased
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=1/4, patience=args.patience, verbose=True,
    )

    writer.add_hparams(args.__dict__, {})
    writer.add_text('model', str(model))

    now = datetime.now()
    # Train the models
    total_step = len(train_loader)
    step = 1
    best_val_acc = 0.0
    for epoch in range(args.num_epochs):
        for i, (video_ids, images, targets) in enumerate(train_loader):
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

            batch_accuracy = float((outputs > 0.0).eq(targets).sum()) / len(targets)

            iteration_time = datetime.now() - now
            # Print log info
            step += 1

            if (i + 1) % args.log_step == 0:
                print_training_info(args, batch_accuracy, epoch, i, iteration_time, loss, step, total_step, writer)
            now = datetime.now()

        # validation step after full epoch
        val_acc = print_validation_info(args, criterion, device, model, val_loader, writer, step)
        lr_scheduler.step(val_acc)
        if val_acc > best_val_acc:
            save_model_checkpoint(args, epoch, model, val_acc, writer.get_logdir())
            best_val_acc = val_acc
    writer.close()


def save_model_checkpoint(args, epoch, model, val_acc, writer_log_dir):
    run_id = writer_log_dir.split('/')[-1]
    model_dir = os.path.join(args.model_path, run_id)
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f'model.pt')
    torch.save(model.state_dict(), model_path)

    model_info = {
        'epoch': epoch,
        'val_acc': val_acc,
        'model_str': str(model)
    }
    json_path = os.path.join(model_dir, 'info.json')
    write_json(model_info, json_path)

    src_model_file = os.path.join(os.path.dirname(__file__), 'model.py')
    dest_model_file = os.path.join(model_dir, 'model.py')
    copy_file(src_model_file, dest_model_file)

    print(f'New checkpoint saved at {model_path}')


def print_training_info(args, batch_accuracy, epoch, i, iteration_time, loss, step, total_step, writer):
    log_info = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}, Iteration time: {}'.format(
        epoch, args.num_epochs, i + 1, total_step, loss.item(), batch_accuracy, iteration_time
    )
    print(log_info)

    writer.add_scalar('training loss', loss.item(), step)
    writer.add_scalar('training acc', batch_accuracy, step)


def print_validation_info(args, criterion, device, model, val_loader, writer, step):
    now = datetime.now()
    model.eval()
    with torch.no_grad():
        loss_values = []
        correct_predictions = 0
        total_predictions = 0

        misclassified_video_ids = set()
        for video_ids, images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            loss_values.append(loss.item())

            predictions = outputs > 0.0
            true_preds = targets.eq(predictions)
            correct_predictions += int(true_preds.sum().cpu())
            misclassified_video_ids.update(video_ids[~true_preds].tolist())
            total_predictions += len(images)
            if args.debug:
                print(outputs)
                print(predictions)
                print(targets)

        val_loss = sum(loss_values) / len(loss_values)
        val_accuracy = correct_predictions / total_predictions
        validation_time = datetime.now() - now

        print(
            'Validation - Loss: {:.3f}, Acc: {:.3f}, Time: {}, Total misclassified videos: {}'
                .format(val_loss, val_accuracy, validation_time, len(misclassified_video_ids))
        )
        writer.add_scalar('validation loss', val_loss, step)
        writer.add_scalar('validation acc', val_accuracy, step)
    return val_accuracy


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
    parser.add_argument('--max_images_per_video', type=int, default=10, help='maximum images to use from one video')
    parser.add_argument('--debug', type=bool, default=False, help='include additional debugging ifo')

    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--regularization', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=22)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--patience', type=int, default=2)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
