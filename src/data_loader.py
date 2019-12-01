import os

import torch
import torch.utils.data as data
from PIL import Image, ImageFile
from torch.utils.data import random_split

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImagesDataset(data.Dataset):
    def __init__(self, original_image_dir, tampered_image_dir, transform=None):
        self.image_paths = []
        self.transform = transform
        self._read_images(original_image_dir, 'original')
        self._read_images(tampered_image_dir, 'tampered')

    def _read_images(self, images_dir, class_name):
        self._read_class_images(class_name, images_dir)

    def _read_class_images(self, class_name, class_images_dir):
        for video_name in os.listdir(class_images_dir):
            for image_name in os.listdir(os.path.join(class_images_dir, video_name)):
                self.image_paths.append({
                    'video_name': video_name,
                    'class': class_name,
                    'img_path': os.path.join(class_images_dir, video_name, image_name)
                })

    def __getitem__(self, index):
        img = self.image_paths[index]

        target = torch.tensor(0.0) if img['class'] == 'original' else torch.tensor(1.0)
        image = Image.open(img['img_path'])
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.image_paths)


def read_dataset(original_image_dir, tampered_image_dir, transform=None, split=0.9):
    full_dataset = ImagesDataset(original_image_dir, tampered_image_dir, transform)

    full_size = len(full_dataset)
    train_size = int(full_size * split)
    val_size = full_size - train_size

    train_dataset, test_dataset = random_split(full_dataset, (train_size, val_size))
    return train_dataset, test_dataset


def get_loader(dataset, batch_size, shuffle, num_workers):
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader
