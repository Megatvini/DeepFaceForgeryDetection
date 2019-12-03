import argparse
from datetime import datetime
import torchvision
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


from data_loader import get_loader, read_dataset
from model import ClassificationCNN


def train(args):
    # show tensorboard graphs with following command: tensorboard --logdir =src/runs
    writer = SummaryWriter()

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    train_dataset, val_dataset = read_dataset(
        args.original_image_dir, args.tampered_image_dir, split=0.90,
        transform=transform, max_images_per_video=args.max_images_per_video
    )

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
    print('train data size: {}, validation data size: {}'.format(len(train_dataset), len(val_dataset)))

    # Build the models
    model = ClassificationCNN().to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.regularization)

    now = datetime.now()
    # Train the models
    total_step = len(train_loader)
    for epoch in range(args.num_epochs):
        model.train()
        for i, (images, targets) in enumerate(train_loader):
            # Set mini-batch dataset
            images = images.to(device)
            targets = targets.to(device)

            # Forward, backward and optimize
            outputs = model(images)
            loss = criterion(outputs, targets)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            grid = torchvision.utils.make_grid(images)
            writer.add_image('images', grid, 0)
            writer.add_graph(model, images)
            writer.close()

            batch_accuracy = float(outputs.round().eq(targets).sum()) / len(targets)

            iteration_time = datetime.now() - now
            # Print log info
            if i % args.log_step == 0:
                log_info = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}, Iteration time: {}'.format(
                    epoch, args.num_epochs, i, total_step, loss.item(), batch_accuracy, iteration_time
                )
                print(log_info)

            now = datetime.now()

          #  if i % args.val_step == 0:
                # validation
        val_loss, val_accuracy = print_validation_info(args, criterion, device, model, val_loader, 'Validation', epoch, writer)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Acc/val', val_accuracy, epoch)
        train_loss, train_accuracy = print_validation_info(args, criterion, device, model, train_loader, 'Training', epoch, writer)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Acc/train', train_accuracy, epoch)



def print_validation_info(args, criterion, device, model, val_loader, mode, epoch, writer):
    #model.eval()
    with torch.no_grad():
        loss_values = []
        correct_predictions = 0
        total_predictions = 0

        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            loss_values.append(loss)

            predictions = outputs.round()
            correct_predictions += int(targets.eq(predictions).sum().cpu())
            total_predictions += len(images)
            if args.debug:
                print(outputs)
                print(predictions)
                print(targets)

        val_loss = sum(loss_values) / len(loss_values)
        val_accuracy = correct_predictions / total_predictions
        print(mode, ' - Loss: {:.3f}, Acc: {:.3f}'.format(val_loss, val_accuracy))
        return val_loss, val_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument(
        '--original_image_dir', type=str, default='/home/jober/Documents/ADL4CV/original_c40/original_sequences/youtube/c40/videos/subset/images',
        help='directory for original images'
    )
    parser.add_argument(
        '--tampered_image_dir', type=str, default='/home/jober/Documents/ADL4CV/NeuralTextures_c40/manipulated_sequences/NeuralTextures/c40/videos/subset/images',
        help='directory for tamprerd images'
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
