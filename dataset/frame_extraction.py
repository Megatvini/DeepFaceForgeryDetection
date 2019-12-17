import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from os.path import join

import cv2
import dlib
from tqdm import tqdm

face_detector = dlib.get_frontal_face_detector()


# generates a quadratic bounding box
# source: https://github.com/ondyari/FaceForensics/blob/master/classification/detect_from_video.py
def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def crop_image(image, face, scale=1.3):
    img_height, img_width, _ = image.shape
    x, y, size = get_boundingbox(face, img_width, img_height, scale=scale)

    # generate cropped image
    cropped_face = image[y:y + size, x:x + size]
    return cropped_face


def extract_face(image):
    detections = face_detector(image, 1)
    if len(detections) == 0:
        return None

    d = detections[0]
    return crop_image(image, d)


def extract_frames(inp):
    data_path, output_path, file_prefix = inp

    os.makedirs(output_path, exist_ok=True)
    reader = cv2.VideoCapture(data_path)
    length = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0
    while reader.isOpened():
        success, image = reader.read()
        if not success:
            break
        face_image = extract_face(image)
        if face_image is not None:
            print(f'no face found, so skipping {output_path}')
            out_path = join(output_path, '{}{:04d}.jpg'.format(file_prefix, frame_num))
            cv2.imwrite(out_path, face_image)
        frame_num += 1
    reader.release()
    return length


def extract_images(videos_path, out_path, num_videos):
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

    executor = ProcessPoolExecutor()
    results = list(tqdm(executor.map(extract_frames, get_video_input_output_pairs()), total=len(video_files)))
    print('total frames extracted: ', sum(results))


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
