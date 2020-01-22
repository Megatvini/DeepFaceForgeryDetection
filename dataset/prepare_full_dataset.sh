#!/bin/bash

echo "YES" | python3 faceforensics_download_v4.py -d NeuralTextures -c c40 /mnt/videos/neural_textures_c40
python3 frame_extraction.py /mnt/videos/neural_textures_c40/manipulated_sequences/NeuralTextures/c40/videos/ /mnt/mtcnn/neural_textures_faces_c40 --num_videos 999999

echo "YES" | python3 faceforensics_download_v4.py -d original -c c40 /mnt/videos/original_c40
python3 frame_extraction.py /mnt/videos/original_c40/original_sequences/youtube/c40/videos/ /mnt/mtcnn/original_faces_c40 --num_videos 999999
