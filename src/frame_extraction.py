import os
from os.path import join
import subprocess
import cv2
from tqdm import tqdm

def extract_frames(data_path, output_path, method='cv2'):
    os.makedirs(output_path, exist_ok=True)
    print('data_path', data_path)
    if method == 'ffmpeg':
        subprocess.check_output(
            'ffmpeg -i {} {}'.format(
                data_path, join(output_path, '%04d.png')),
            shell=True, stderr=subprocess.STDOUT)
    elif method == 'cv2':
        reader = cv2.VideoCapture(data_path)
        frame_num = 0
        while reader.isOpened():
            success, image = reader.read()
            if not success:
                break
            cv2.imwrite(join(output_path, '{:04d}.png'.format(frame_num)),
                        image)
            print('frame num', frame_num)
            frame_num += 1
        reader.release()
    else:
        raise Exception('Wrong extract frames method: {}'.format(method))


def extract_method_videos(data_path):
    images_path = join(data_path, 'images')
    for video in tqdm(os.listdir(data_path)):
        image_folder = video.split('.')[0]
        extract_frames(join(data_path, video),
                       join(images_path, image_folder))


if __name__ == '__main__':
    extract_method_videos('/home/jober/Documents/GR/Extraction_test')