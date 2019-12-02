#!/bin/bash

python faceforensics_download_v4.py -d NeuralTextures -c c40 neural_textures_c40
python faceforensics_download_v4.py -d original -c c40 original_c40

python frame_extraction.py original_c40/original_sequences/youtube/c40/videos/ images_large/original/ --num_videos 999999
python frame_extraction.py neural_textures_c40/manipulated_sequences/NeuralTextures/c40/videos/ images_large/neural_textures/ --num_videos 999999
