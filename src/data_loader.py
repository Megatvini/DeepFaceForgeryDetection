import os

import torch
import torch.utils.data as data
from PIL import Image, ImageFile


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
        image = self.transform(Image.open(img['img_path']))
        return image, target

    def __len__(self):
        return len(self.image_paths)


def get_loader(original_image_dir, tampered_image_dir, transform, batch_size, shuffle, num_workers):
    dataset = ImagesDataset(original_image_dir, tampered_image_dir, transform)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader
