import argparse
import os
from os.path import join

import cv2
from tqdm import tqdm


def extract_frames(data_path, output_path, file_prefix):
    os.makedirs(output_path, exist_ok=True)
    reader = cv2.VideoCapture(data_path)
    length = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=length) as p_bar:
        frame_num = 0
        while reader.isOpened():
            success, image = reader.read()
            if not success:
                break
            cv2.imwrite(join(output_path, '{}{:04d}.png'.format(file_prefix, frame_num)), image)
            frame_num += 1
            p_bar.update(1)
        reader.release()


def extract_images(videos_path, out_path, num_videos):
    print('extracting video frames from {} to {}'.format(videos_path, out_path))

    video_files = os.listdir(videos_path)
    print('total videos found - {}, extracting from - {}'.format(len(video_files), min(len(video_files), num_videos)))

    for index, video_file in enumerate(video_files[:num_videos]):
        video_file_name = video_file.split('.')[0]
        video_out_path = os.path.join(out_path, video_file_name)
        video_path = os.path.join(videos_path, video_file)
        print('extracting for video# {}'.format(index + 1))
        file_prefix = '{}_'.format(video_file_name)
        extract_frames(video_path, video_out_path, file_prefix=file_prefix)


def parse_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('path_to_videos', type=str, help='path for input videos')
    args_parser.add_argument('output_path', type=str, help='output images path')
    args_parser.add_argument('--num_videos', type=int, default=10, help='number of videos to extract images from')
    args = args_parser.parse_args()
    return args


def main():
    args = parse_args()
    videos_path = args.path_to_videos
    out_path = args.output_path
    num_videos = args.num_videos
    extract_images(videos_path, out_path, num_videos)


if __name__ == '__main__':
    main()
