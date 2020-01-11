import os
import random

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImagesDataset(Dataset):
    def __init__(
            self, original_video_dirs, tampered_video_dirs, max_images_per_video,
            max_videos=1000, transform=None, window_size=5
    ):
        self.max_images_per_video = max_images_per_video
        self.max_videos = max_videos
        self.image_paths = []
        self.transform = transform
        self.window_size = window_size
        self._read_images(original_video_dirs, 'original')
        self._read_images(tampered_video_dirs, 'tampered')
        self.image_paths = sorted(self.image_paths, key=lambda x: x['img_path'])

    def _read_images(self, video_dirs, class_name):
        for video_dir in video_dirs[:self.max_videos]:
            self._read_class_images(class_name, video_dir)

    def _read_class_images(self, class_name, video_dir):
        video_id = get_file_name(video_dir)
        sorted_images_names = sorted(os.listdir(video_dir))[:self.max_images_per_video]
        for image_name in sorted_images_names:
            self.image_paths.append({
                'video_id': int(video_id),
                'class': class_name,
                'img_path': os.path.join(video_dir, image_name)
            })

    def __getitem__(self, index):
        data = [self._get_item(index + i) for i in range(-self.window_size//2 + 1, self.window_size//2 + 1)]
        mid_video_id, mid_image, target = data[len(data)//2]
        images = [x[1] if x[0] == mid_video_id else mid_image for x in data]
        if self.window_size > 1:
            return torch.tensor(mid_video_id), torch.stack(images).permute(1, 0, 2, 3), target
        else:
            return mid_video_id, images[0], target

    def _get_item(self, index):
        img = self.image_paths[index]
        target = torch.tensor(0.0) if img['class'] == 'original' else torch.tensor(1.0)
        image = Image.open(img['img_path'])
        if self.transform is not None:
            image = self.transform(image)
        return img['video_id'], image, target

    def __len__(self):
        return len(self.image_paths) - self.window_size // 2 - 1


def listdir_with_full_paths(dir_path):
    return [os.path.join(dir_path, x) for x in os.listdir(dir_path)]


def random_split(data, split):
    size = int(len(data)*split)
    random.shuffle(data)
    return data[:size], data[size:]


def get_file_name(file_path):
    return file_path.split('/')[-1]


def read_json(file_path):
    with open(file_path) as inp:
        return json.load(inp)


def get_sets(data):
    return {x[0] for x in data} | {x[1] for x in data} | {'_'.join(x) for x in data}


def get_video_ids(spl, splits_path):
    return get_sets(read_json(os.path.join(splits_path, f'{spl}.json')))


def read_dataset(
        original_data_dir, tampered_data_dir, transform=None, max_videos=1000, window_size=5,
        max_images_per_video=40, splits_path='../dataset/splits/'
):
    original_video_dir_paths = listdir_with_full_paths(original_data_dir)
    tampered_video_dir_paths = listdir_with_full_paths(tampered_data_dir)

    train_video_ids = get_video_ids('train', splits_path)
    train_videos_original = [x for x in original_video_dir_paths if get_file_name(x) in train_video_ids]
    train_videos_tampered = [x for x in tampered_video_dir_paths if get_file_name(x) in train_video_ids]

    train_dataset = ImagesDataset(
        train_videos_original, train_videos_tampered, max_images_per_video=max_images_per_video, transform=transform,
        max_videos=max_videos, window_size=window_size
    )

    val_video_ids = get_video_ids('val', splits_path)
    val_videos_original = [x for x in original_video_dir_paths if get_file_name(x) in val_video_ids]
    val_videos_tampered = [x for x in tampered_video_dir_paths if get_file_name(x) in val_video_ids]
    val_dataset = ImagesDataset(
        val_videos_original, val_videos_tampered, max_images_per_video=max_images_per_video, transform=transform,
        max_videos=max_videos, window_size=window_size
    )

    return train_dataset, val_dataset


def get_loader(dataset, batch_size, shuffle, num_workers):
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              drop_last=True,
                                              pin_memory=True)
    return data_loader
