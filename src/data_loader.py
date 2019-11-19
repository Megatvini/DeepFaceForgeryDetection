import os

import torch
import torch.utils.data as data
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImagesDataset(data.Dataset):
    def __init__(self, images_dir, transform=None):
        self.images = []
        self.transform = transform
        self._read_images(images_dir)

    def _read_images(self, images_dir):
        originals = os.path.join(images_dir, 'original')
        for file_path in os.listdir(originals):
            img = Image.open(os.path.join(originals, file_path))
            self.images.append({
                'class': 'original',
                'img': img
            })

        tampered = os.path.join(images_dir, 'tampered')
        for file_path in os.listdir(tampered):
            img = Image.open(os.path.join(tampered, file_path))
            self.images.append({
                'class': 'tampered',
                'img': img
            })

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        img = self.images[index]

        target = 0 if img['class'] == 'original' else 1
        image = self.transform(img['img'])
        return image, target

    def __len__(self):
        return len(self.images)


def get_loader(images_dir, transform, batch_size, shuffle, num_workers):
    dataset = ImagesDataset(images_dir, transform)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader
