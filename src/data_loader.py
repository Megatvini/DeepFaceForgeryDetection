import json
import os
import random

import torch
from PIL import Image, ImageFile
from torch import tensor
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CompositeDataset(Dataset):
    def __init__(self, *datasets) -> None:
        super().__init__()
        self.datasets = datasets

    def __getitem__(self, index: int):
        for d in self.datasets:
            if index < len(d):
                return d[index]
            index -= len(d)

    def __len__(self) -> int:
        return sum(map(len, self.datasets))


class ImagesDataset(Dataset):
    def __init__(self, video_dirs, name, target, max_images_per_video, max_videos, transform, window_size):
        self.name = name
        self.target = torch.tensor(target).float()
        self.max_images_per_video = max_images_per_video
        self.max_videos = max_videos
        self.image_paths = []
        self.transform = transform
        self.window_size = window_size
        self._read_images(video_dirs, name)
        self.image_paths = sorted(self.image_paths, key=lambda x: x['img_path'])

    def _read_images(self, video_dirs, class_name):
        for video_dir in video_dirs[:self.max_videos]:
            self._read_class_images(class_name, video_dir)

    def _read_class_images(self, class_name, video_dir):
        video_id = get_file_name(video_dir)
        sorted_images_names = sorted(os.listdir(video_dir))[:self.max_images_per_video]
        for image_name in sorted_images_names:
            frame_id = image_name.split('_')[-1].split('.')[0]
            self.image_paths.append({
                'video_id': video_id,
                'frame_id': frame_id,
                'class': class_name,
                'img_path': os.path.join(video_dir, image_name)
            })

    def __getitem__(self, index):
        data = [self._get_item(index + i) for i in range(-self.window_size//2 + 1, self.window_size//2 + 1)]
        mid_video_id, mid_frame_id, mid_image, target = data[len(data)//2]
        images = [x[2] if x[0] == mid_video_id else mid_image for x in data]
        if self.window_size > 1:
            return mid_video_id, mid_frame_id, torch.stack(images).permute(1, 0, 2, 3), target
        else:
            image_tensor = images[0]
            return mid_video_id, mid_frame_id, image_tensor, target

    def _get_item(self, index):
        img = self.image_paths[index]
        target = self.target
        image = Image.open(img['img_path'])
        if self.transform is not None:
            image = self.transform(image)
        return img['video_id'], img['frame_id'], image, target

    def __len__(self):
        return len(self.image_paths) - self.window_size // 2


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
    return {x[0] for x in data} | {x[1] for x in data} | {'_'.join(x) for x in data} | {'_'.join(x[::-1]) for x in data}


def get_video_ids(spl, splits_path):
    return get_sets(read_json(os.path.join(splits_path, f'{spl}.json')))


def read_train_test_val_dataset(
        dataset_dir, name, target, splits_path, **dataset_kwargs
):
    for spl in ['train', 'val', 'test']:
        video_ids = get_video_ids(spl, splits_path)
        video_paths = listdir_with_full_paths(dataset_dir)
        videos = [x for x in video_paths if get_file_name(x) in video_ids]
        dataset = ImagesDataset(videos, name, target, **dataset_kwargs)
        yield dataset


def read_dataset(
        data_dir, transform, max_videos, window_size,
        max_images_per_video, splits_path='../dataset/splits/'
):
    data_class_dirs = os.listdir(data_dir)
    data_sets = {}
    for data_class_dir in data_class_dirs:
        data_class_dir_path = os.path.join(data_dir, data_class_dir)
        target = 0 if 'original' in data_class_dir.lower() else 1
        data_sets[data_class_dir] = read_train_test_val_dataset(
            data_class_dir_path, data_class_dir, target, splits_path, transform=transform,
            max_videos=max_videos, max_images_per_video=max_images_per_video, window_size=window_size
        )
    return data_sets


def get_loader(dataset, batch_size, shuffle, num_workers, drop_last=True):
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              drop_last=drop_last,
                                              pin_memory=True)
    return data_loader
