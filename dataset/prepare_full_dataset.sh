#!/bin/bash


echo "YES" | python3 faceforensics_download_v4.py -d original -c c40 $1/videos/original_c40 && \
python3 frame_extraction.py $1/videos/original_c40/original_sequences/youtube/c40/videos/ $1/mtcnn/original_faces_c40 --num_videos 999999

echo "YES" | python3 faceforensics_download_v4.py -d original -c c23 $1/videos/original_c23 && \
python3 frame_extraction.py $1/videos/original_c23/original_sequences/youtube/c23/videos/ $1/mtcnn/original_faces_c23 --num_videos 999999

echo "YES" | python3 faceforensics_download_v4.py -d original -c raw $1/videos/original_raw && \
python3 frame_extraction.py $1/videos/original_raw/original_sequences/youtube/raw/videos/ $1/mtcnn/original_faces_raw --num_videos 999999


echo "YES" | python3 faceforensics_download_v4.py -d NeuralTextures -c c40 $1/videos/neural_textures_c40 && \
python3 frame_extraction.py $1/videos/neural_textures_c40/manipulated_sequences/NeuralTextures/c40/videos/ $1/mtcnn/neural_textures_faces_c40 --num_videos 999999

echo "YES" | python3 faceforensics_download_v4.py -d NeuralTextures -c c23 $1/videos/neural_textures_c23 && \
python3 frame_extraction.py $1/videos/neural_textures_c23/manipulated_sequences/NeuralTextures/c23/videos/ $1/mtcnn/neural_textures_faces_c23 --num_videos 999999

echo "YES" | python3 faceforensics_download_v4.py -d NeuralTextures -c raw $1/videos/neural_textures_raw && \
python3 frame_extraction.py $1/videos/neural_textures_raw/manipulated_sequences/NeuralTextures/raw/videos/ $1/mtcnn/neural_textures_faces_raw --num_videos 999999


echo "YES" | python3 faceforensics_download_v4.py -d Deepfakes -c c40 $1/videos/deepfakes_c40 && \
python3 frame_extraction.py $1/videos/deepfakes_c40/manipulated_sequences/Deepfakes/c40/videos/ $1/mtcnn/deepfakes_faces_c40 --num_videos 999999

echo "YES" | python3 faceforensics_download_v4.py -d Deepfakes -c c23 $1/videos/deepfakes_c23 && \
python3 frame_extraction.py $1/videos/deepfakes_c23/manipulated_sequences/Deepfakes/c23/videos/ $1/mtcnn/deepfakes_faces_c23 --num_videos 999999

echo "YES" | python3 faceforensics_download_v4.py -d Deepfakes -c raw $1/videos/deepfakes_raw && \
python3 frame_extraction.py $1/videos/deepfakes_raw/manipulated_sequences/Deepfakes/raw/videos/ $1/mtcnn/deepfakes_faces_raw --num_videos 999999


echo "YES" | python3 faceforensics_download_v4.py -d Face2Face -c c40 $1/videos/face2face_c40 && \
python3 frame_extraction.py $1/videos/face2face_c40/manipulated_sequences/Face2Face/c40/videos/ $1/mtcnn/face2face_faces_c40 --num_videos 999999

echo "YES" | python3 faceforensics_download_v4.py -d Face2Face -c c23 $1/videos/face2face_c23 && \
python3 frame_extraction.py $1/videos/face2face_c23/manipulated_sequences/Face2Face/c23/videos/ $1/mtcnn/face2face_faces_c23 --num_videos 999999

echo "YES" | python3 faceforensics_download_v4.py -d Face2Face -c raw $1/videos/face2face_raw && \
python3 frame_extraction.py $1/videos/face2face_raw/manipulated_sequences/Face2Face/raw/videos/ $1/mtcnn/face2face_faces_raw --num_videos 999999


echo "YES" | python3 faceforensics_download_v4.py -d FaceSwap -c c40 $1/videos/faceswap_c40 && \
python3 frame_extraction.py $1/videos/faceswap_c40/manipulated_sequences/FaceSwap/c40/videos/ $1/mtcnn/faceswap_faces_c40 --num_videos 999999

echo "YES" | python3 faceforensics_download_v4.py -d FaceSwap -c c23 $1/videos/faceswap_c23 && \
python3 frame_extraction.py $1/videos/faceswap_c23/manipulated_sequences/FaceSwap/c23/videos/ $1/mtcnn/faceswap_faces_c23 --num_videos 999999

echo "YES" | python3 faceforensics_download_v4.py -d FaceSwap -c raw $1/videos/faceswap_raw && \
python3 frame_extraction.py $1/videos/faceswap_raw/manipulated_sequences/FaceSwap/raw/videos/ $1/mtcnn/faceswap_faces_raw --num_videos 999999
