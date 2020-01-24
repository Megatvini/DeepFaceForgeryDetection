import argparse
import os
from os.path import join

import cv2
import mmcv
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm


def extract_frames(face_detector, data_path, output_path, file_prefix):
    os.makedirs(output_path, exist_ok=True)
    video = mmcv.VideoReader(data_path)
    length = video.frame_cnt
    for frame_num, frame in enumerate(video):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        out_file_path = join(output_path, '{}{:04d}.png'.format(file_prefix, frame_num))
        if not os.path.exists(out_file_path):
            face_detector(image, save_path=out_file_path)
    return length


def extract_images(device, videos_path, out_path, num_videos):
    print('extracting video frames from {} to {}'.format(videos_path, out_path))

    video_files = os.listdir(videos_path)
    print('total videos found - {}, extracting from - {}'.format(len(video_files), min(len(video_files), num_videos)))
    video_files = video_files[:num_videos]

    def get_video_input_output_pairs():
        for index, video_file in enumerate(video_files):
            video_file_name = video_file.split('.')[0]
            v_out_path = os.path.join(out_path, video_file_name)
            v_path = os.path.join(videos_path, video_file)
            f_prefix = '{}_'.format(video_file_name)
            yield v_path, v_out_path, f_prefix

    face_detector = MTCNN(device=device, margin=16)
    face_detector.eval()

    for data_path, output_path, file_prefix in tqdm(get_video_input_output_pairs(), total=len(video_files)):
        extract_frames(face_detector, data_path, output_path, file_prefix)


def parse_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('path_to_videos', type=str, help='path for input videos')
    args_parser.add_argument('output_path', type=str, help='output images path')
    args_parser.add_argument('--num_videos', type=int, default=10, help='number of videos to extract images from')
    args = args_parser.parse_args()
    return args


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    args = parse_args()
    videos_path = args.path_to_videos
    out_path = args.output_path
    num_videos = args.num_videos
    extract_images(device, videos_path, out_path, num_videos)


if __name__ == '__main__':
    main()
