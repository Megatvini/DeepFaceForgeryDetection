import os

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImagesDataset(Dataset):
    def __init__(self, original_video_dirs, tampered_video_dirs, transform=None):
        self.image_paths = []
        self.transform = transform
        self._read_images(original_video_dirs, 'original')
        self._read_images(tampered_video_dirs, 'tampered')

    def _read_images(self, video_dirs, class_name):
        for video_dir in video_dirs:
            self._read_class_images(class_name, video_dir)

    def _read_class_images(self, class_name, video_dir):
        video_id = video_dir.split('/')[-1]
        for image_name in os.listdir(video_dir):
            self.image_paths.append({
                'video_id': video_id,
                'class': class_name,
                'img_path': os.path.join(video_dir, image_name)
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


def listdir_with_full_paths(dir_path):
    return [os.path.join(dir_path, x) for x in os.listdir(dir_path)]


def random_split(data, split=0.9):
    size = int(len(data)*split)
    random.shuffle(data)
    return data[:size], data[size:]


def read_dataset(original_data_dir, tampered_data_dir, transform=None, split=0.9):
    original_video_dir_paths = listdir_with_full_paths(original_data_dir)
    tampered_video_dir_paths = listdir_with_full_paths(tampered_data_dir)

    # need to make sure tampered videos in validation don't contain any videos from training
    train_videos_original, val_videos_original = random_split(original_video_dir_paths, split)
    train_videos_tampered, val_videos_tampered = random_split(tampered_video_dir_paths, split)

    train_dataset = ImagesDataset(train_videos_original, train_videos_tampered, transform)
    val_dataset = ImagesDataset(val_videos_original, val_videos_tampered, transform)
    return train_dataset, val_dataset


def get_loader(dataset, batch_size, shuffle, num_workers):
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader
