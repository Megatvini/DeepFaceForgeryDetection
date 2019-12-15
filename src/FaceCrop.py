import os
import argparse
import cv2
import dlib
from tqdm import tqdm

"""
    Function facecrop automatically extracts face region for all frames and saves the cropped image 
    - Assumes current file structure of one folder per video containing all extracted frames
    - Drastically reduces the size of the data we need to store 
"""


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


# generates cropped image containing only the face region for all frames
# available in input path and saves resulting image to output path

def facecrop(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    face_detector = dlib.get_frontal_face_detector()
    # iterate overall folders in input path containing drames
    for foldername in os.listdir(input_path):
        framepath = os.path.join(input_path, foldername)
        # save cropped image in folderof same name in ouput path
        output_folder = os.path.join(output_path, foldername)
        os.makedirs(output_folder, exist_ok=True)
        print(framepath)
        length = len(os.listdir(framepath))
        with tqdm(total=length) as p_bar:
            # iterate over all files in the current folder
            for filename in os.listdir(framepath):
                # go to the next file if the current file is not a png-image
                if filename[-3:] == 'png':
                    filepath = os.path.join(framepath, filename)
                    # read current frame
                    im = cv2.imread(filepath)
                    height, width = im.shape[:2]
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    faces = face_detector(gray, 1)
                    if len(faces):
                        # For now only take biggest face
                        face = faces[0]
                        x, y, size = get_boundingbox(face, width, height)
                        # generate cropped image
                        cropped_face = im[y:y + size, x:x + size]
                    # save cropped image in the desired location
                    im_out_path = os.path.join(output_folder, filename)
                    cv2.imwrite(im_out_path, cropped_face)
                p_bar.update(1)
    pass


# assert that a cropped image has been generated for each frame of every available video
def checkfacecropping(input_path, ouput_path):
    nr_input_videos = len(os.listdir(input_path))
    nr_output_videos = len(os.listdir(output_path))
    oklist = []
    notoklist = []
    # check whether the number of video folders in the input path matches the number of folders in the ouput path
    if nr_input_videos == nr_output_videos:
        print('All videos have been cropped.')
    else:
        print(nr_input_videos - nr_output_videos, 'videos have not been cropped.')
    # for each video folder in the output path check whether all frames have been cropped
    for out_folder in os.listdir(output_path):
        out_frame_path = os.path.join(output_path, out_folder)
        in_frame_path = os.path.join(input_path, out_folder)
        nr_input_frames = len(os.listdir(in_frame_path))
        nr_output_frames = len(os.listdir(out_frame_path))
        # if all frames have been cropped append the number of the current video to oklist
        if nr_input_frames == nr_output_frames:
            oklist.append(out_folder)
        # if not all frames have been cropped append the number of the current video to notoklist
        else:
            notoklist.append(out_folder)
    # return list of all videos for which all frames have been cropped and list
    # of videos for which only a subset of frames have been cropped
    return oklist, notoklist


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input_path', '-i', type=str,
                   default='/home/jober/Documents/GR/original_sequences/c0/videos/images')
    p.add_argument('--output_path', '-o', type=str,
                   default='/home/jober/Documents/GR/original_sequences/c0/images')
    args = p.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    # facecrop(input_path, output_path)

    oklist, notoklist = checkfacecropping(input_path, output_path)