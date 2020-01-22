#!/bin/bash

echo "YES" | python faceforensics_download_v4.py -d NeuralTextures -c c40 neural_textures_c40
python frame_extraction.py original_c40/original_sequences/youtube/c40/videos/ /mnt/mtcnn/neural_textures_faces_c40 --num_videos 999999

echo "YES" | python faceforensics_download_v4.py -d original -c c40 original_c40
python frame_extraction.py neural_textures_c40/manipulated_sequences/NeuralTextures/c40/videos/ /mnt/original_faces_c40 --num_videos 999999
