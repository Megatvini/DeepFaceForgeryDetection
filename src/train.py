import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from facenet_pytorch import fixed_image_standardization
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from data_loader import get_loader, read_dataset, CompositeDataset
from model import FaceRecognitionCNN
from utils import write_json, copy_file, count_parameters


def read_training_dataset(args, transform):
    datasets = read_dataset(
        args.data_dir, transform=transform,
        max_images_per_video=args.max_images_per_video, max_videos=args.max_videos,
        window_size=args.window_size, splits_path=args.splits_path
    )
    # only neural textures c40 and original c40
    datasets = {
        k: v for k, v in datasets.items() 
        if ('original' in k or 'neural' in k) and 'c40' in k
    }
    print('Using training data: ')
    print('\n'.join(sorted(datasets.keys())))

    trains, vals, tests = [], [], []
    for data_dir_name, dataset in datasets.items():
        train, val, test = dataset
        # repeat original data multiple times to balance out training data
        compression = data_dir_name.split('_')[-1]
        num_tampered_with_same_compression = len({x for x in datasets.keys() if compression in x}) - 1
        count = 1 if 'original' not in data_dir_name else num_tampered_with_same_compression
        for _ in range(count):
            trains.append(train)
        vals.append(val)
        tests.append(test)
    return CompositeDataset(*trains), CompositeDataset(*vals), CompositeDataset(*tests)


def run_train(args):
    # show tensorboard graphs with following command: tensorboard --logdir=src/runs
    writer = SummaryWriter(comment=args.comment)

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

    train_dataset, val_dataset, test_dataset = read_training_dataset(args, transform)

    tqdm.write('train data size: {}, validation data size: {}'.format(len(train_dataset), len(val_dataset)))

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
    model = FaceRecognitionCNN().to(device)
    if args.freeze_first_epoch:
        for m in model.resnet.parameters():
            m.requires_grad_(False)

    input_shape = next(iter(train_loader))[2].shape
    print('input shape', input_shape)
    # need to call this before summary!!!
    model.eval()
    # summary(model, input_shape[1:], batch_size=input_shape[0], device=device)
    print('model params (trainable, total):', count_parameters(model))

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

    # Train the models
    total_step = len(train_loader)
    step = 1
    best_val_acc = 0.5
    for epoch in range(args.num_epochs):
        for i, (video_ids, frame_ids, images, targets) in \
                tqdm(enumerate(train_loader), desc=f'training epoch {epoch}', total=len(train_loader)):
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

            # Print log info
            step += 1

            if (i + 1) % args.log_step == 0:
                print_training_info(batch_accuracy, loss, step, writer)

            if (i + 1) % args.val_step == 0:
                val_acc, pr_acc, tmp_acc = print_validation_info(
                    args, criterion, device, model, val_loader, writer, epoch, step
                )
                if val_acc > best_val_acc:
                    save_model_checkpoint(args, epoch, model, (val_acc, pr_acc, tmp_acc), writer.get_logdir())
                    best_val_acc = val_acc

        # validation step after full epoch
        val_acc, pr_acc, tmp_acc = print_validation_info(
            args, criterion, device, model, val_loader, writer, epoch, step
        )
        lr_scheduler.step(val_acc)
        if val_acc > best_val_acc:
            save_model_checkpoint(args, epoch, model, (val_acc, pr_acc, tmp_acc), writer.get_logdir())
            best_val_acc = val_acc

        if args.freeze_first_epoch and epoch == 0:
            for m in model.resnet.parameters():
                m.requires_grad_(True)
            tqdm.write('Fine tuning on')

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

    tqdm.write(f'New checkpoint saved at {model_path}')


def print_training_info(batch_accuracy, loss, step, writer):
    log_info = 'Training - Loss: {:.4f}, Accuracy: {:.4f}'.format(loss.item(), batch_accuracy)
    tqdm.write(log_info)

    writer.add_scalar('training loss', loss.item(), step)
    writer.add_scalar('training acc', batch_accuracy, step)


def print_validation_info(args, criterion, device, model, val_loader, writer, epoch, step):
    model.eval()
    with torch.no_grad():
        loss_values = []
        all_predictions = []
        all_targets = []
        for video_ids, frame_ids, images, targets in tqdm(val_loader, desc=f'validation ep. {epoch}'):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            loss_values.append(loss.item())

            predictions = outputs > 0.0
            all_predictions.append(predictions)
            all_targets.append(targets)
            if args.debug:
                tqdm.write(outputs)
                tqdm.write(predictions)
                tqdm.write(targets)

        val_loss = sum(loss_values) / len(loss_values)

        all_predictions = torch.cat(all_predictions).int()
        all_targets = torch.cat(all_targets).int()
        val_accuracy = (all_predictions == all_targets).sum().float().item() / all_targets.shape[0]

        total_target_tampered = all_targets.sum().float().item()
        tampered_accuracy = (all_predictions * all_targets).sum().item() / total_target_tampered

        total_target_pristine = (1 - all_targets).sum().float().item()
        pristine_accuracy = (1 - (all_predictions | all_targets)).sum().item() / total_target_pristine

        tqdm.write(
            'Validation - Loss: {:.3f}, Acc: {:.3f}, Prs: {:.3f}, Tmp: {:.3f}'.format(
                val_loss, val_accuracy, pristine_accuracy, tampered_accuracy
            )
        )
        writer.add_scalar('validation loss', val_loss, step)
        writer.add_scalar('validation acc', val_accuracy, step)
        writer.add_scalar('pristine acc', pristine_accuracy, step)
        writer.add_scalar('tampered acc', tampered_accuracy, step)
    return val_accuracy, pristine_accuracy, tampered_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--data_dir', type=str, default='../dataset/images_tiny', help='directory for data with images')
    parser.add_argument('--log_step', type=int, default=10, help='step size for printing training log info')
    parser.add_argument('--val_step', type=int, default=100, help='step size for validation during epoch')
    parser.add_argument('--max_images_per_video', type=int, default=10, help='maximum images to use from one video')
    parser.add_argument('--debug', type=bool, default=False, help='include additional debugging ifo')

    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--regularization', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=22)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--window_size', type=int, default=1)
    parser.add_argument('--max_videos', type=int, default=1000)
    parser.add_argument('--splits_path', type=str, default='../dataset/splits/')
    parser.add_argument('--encoder_model_path', type=str, default='models/Jan12_10-57-19_gpu-training/model.pt')
    parser.add_argument('--freeze_first_epoch', type=bool, default=False)
    parser.add_argument('--comment', type=str, default='')
    args = parser.parse_args()
    run_train(args)


if __name__ == '__main__':
    main()
