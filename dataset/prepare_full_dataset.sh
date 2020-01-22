#!/bin/bash


echo "YES" | python3 faceforensics_download_v4.py -d original -c c40 /mnt/videos/original_c40 && \
python3 frame_extraction.py /mnt/videos/original_c40/original_sequences/youtube/c40/videos/ /mnt/mtcnn/original_faces_c40 --num_videos 999999

echo "YES" | python3 faceforensics_download_v4.py -d original -c c23 /mnt/videos/original_c23 && \
python3 frame_extraction.py /mnt/videos/original_c23/original_sequences/youtube/c23/videos/ /mnt/mtcnn/original_faces_c23 --num_videos 999999

echo "YES" | python3 faceforensics_download_v4.py -d original -c raw /mnt/videos/original_raw && \
python3 frame_extraction.py /mnt/videos/original_raw/original_sequences/youtube/raw/videos/ /mnt/mtcnn/original_faces_raw --num_videos 999999


echo "YES" | python3 faceforensics_download_v4.py -d NeuralTextures -c c40 /mnt/videos/neural_textures_c40 && \
python3 frame_extraction.py /mnt/videos/neural_textures_c40/manipulated_sequences/NeuralTextures/c40/videos/ /mnt/mtcnn/neural_textures_faces_c40 --num_videos 999999

echo "YES" | python3 faceforensics_download_v4.py -d NeuralTextures -c c23 /mnt/videos/neural_textures_c23 && \
python3 frame_extraction.py /mnt/videos/neural_textures_c23/manipulated_sequences/NeuralTextures/c23/videos/ /mnt/mtcnn/neural_textures_faces_c23 --num_videos 999999

echo "YES" | python3 faceforensics_download_v4.py -d NeuralTextures -c raw /mnt/videos/neural_textures_raw && \
python3 frame_extraction.py /mnt/videos/neural_textures_raw/manipulated_sequences/NeuralTextures/raw/videos/ /mnt/mtcnn/neural_textures_faces_raw --num_videos 999999
