"""
Download video from YouTube

Usage:
URL=YOUTUBE_VIDEO_URL
PATH_TO_VID=path/to/video.mp4
python download_video.py --url URL --output PATH_TO_VID
"""
import os
import argparse
from pytube import YouTube

# ==== Argument Parser ==== #
parser = argparse.ArgumentParser(description='Argparser for download_video.py')
parser.add_argument('--url', dest='url', help='Youtube video path')
parser.add_argument('--output', dest='output', help='Video download location')
args = parser.parse_args()

if __name__ == "__main__":
    file_path, filename = os.path.split(args.output)
    filename = filename.split('.')[0]

    YouTube(args.url).\
        streams.\
        filter(adaptive=True, res="720p").\
        first().\
        download(output_path = file_path, filename = filename)