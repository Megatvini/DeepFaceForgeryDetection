import os

import torch
import torch.utils.data as data
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImagesDataset(data.Dataset):
    def __init__(self, images_dir, transform=None):
        self.image_paths = []
        self.transform = transform
        self._read_images(images_dir)

    def _read_images(self, images_dir):
        self._read_class_images('original', images_dir)
        self._read_class_images('tampered', images_dir)

    def _read_class_images(self, class_name, images_dir):
        class_images_dir = os.path.join(images_dir, class_name)
        for file_path in os.listdir(class_images_dir):
            self.image_paths.append({
                'class': class_name,
                'img_path': os.path.join(class_images_dir, file_path)
            })

    def __getitem__(self, index):
        img = self.image_paths[index]

        target = torch.tensor(0.0) if img['class'] == 'original' else torch.tensor(1.0)
        image = self.transform(Image.open(img['img_path']))
        return image, target

    def __len__(self):
        return len(self.image_paths)


def get_loader(images_dir, transform, batch_size, shuffle, num_workers):
    dataset = ImagesDataset(images_dir, transform)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader
