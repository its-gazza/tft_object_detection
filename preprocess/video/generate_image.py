"""
Generate images for input video

This script will capture images from given video per x second.

Usage:
VID_PATH=/path/to/vid.mp4
OUT_PATH=/path/to/folder
INTERVAL=1
python generate_image.py --video VID_PATH --output_dir OUT_PATH --interval INTERVAL --showvideo
"""
import cv2
import os
import argparse
import logging

# ==== Argument Parser ==== # 
parser = argparse.ArgumentParser(description='Argparser for generate_image.py')
parser.add_argument('-v', '--video', dest='vid', help='Path to video')
parser.add_argument('--output_dir', dest='output_path', 
                    help='Path to image locaiton, will create folder if doesn\'t exist')
parser.add_argument('--interval', dest='interval', default=1, type=int,
                    help='interval between image, unit based on seconds')
parser.add_argument('--showvideo', dest='show_video', action='store_true')
args = parser.parse_args()

# ==== Functions ==== # 
def createDirectory(path_loc):
    """Create directory"""
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        logging.warning(f"Path not found, created path: {args.output_path}")
    else:
        pass
        logging.info(f"Path found, skipping createDirectory")

def generate_image(video_path, output_path, interval, show_video=True):
    """Generate Image from Video

    Args:
        video_path (str): Video Path
        output_path (str): Image Path
        interval (int): Skip fram per x second
        show_video (bool, optional): Show video on screen. Defaults to True.

    Raises:
        FileNotFoundError: If video_path is not found, return error
    """
    _, filename = os.path.split(video_path)
    video_name = filename.split('.')[0]
    logging.info(f"Path: {video_path}, Video name: {video_name}")
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    totl_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
    # Sanity check
    if not os.path.exists(args.vid):
        raise FileNotFoundError(f"Cannot find file in :{args.vid}")

    # Parameters
    count = 0
    img_num = 0

    # Open and loop video
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            # Write image to location
            img_path = os.path.join(output_path, f'{video_name}_{img_num}.jpg')
            cv2.imwrite(img_path, frame)

            # Increase parameter
            count += (fps * interval)
            img_num += 1
            video.set(1, count)

            logging.info(f"{int(count)}/{int(totl_frame)} ({int(100*count/totl_frame)}%) Done")

            if show_video:
                cv2.imshow("Video", frame)

            if cv2.waitKey(1) and 0xFF == ord('q'):
                break

        else:
            video.release()
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    createDirectory(args.output_path)
    generate_image(args.vid, args.output_path, args.interval, args.show_video)