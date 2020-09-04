import io
import re
import os
import cv2
import scipy.misc
import numpy as np
import six
import time
from six import BytesIO
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils


# Set memory growth to True so it won't crash
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Video and model path
model_path = 'exported-models/mobile_net/saved_model/'
video_path = 'data/vid_1/vid.mp4'

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path (this can be local or on colossus)

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

category_index = {
    1: {'id': 1, 'name': 'player'},
    2: {'id': 2, 'name': 'trait'}
}

# Load model
tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load(model_path)

vid = cv2.VideoCapture(video_path)
count = 0
img_count = 0

while vid.isOpened():
    done = False
    ret, frame = vid.read()
    if ret:
      if done:
        vid.release()
        break

      input_tensor = np.expand_dims(frame, 0)
      detections = detect_fn(input_tensor)

      label_id_offset = 1
      image_np_with_detections = frame.copy()
      viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'][0].numpy(),
          detections['detection_classes'][0].numpy().astype(np.int32),
          detections['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.6,
          agnostic_mode=False)

      cv2.imshow("Frame", image_np_with_detections)
      cv2.imwrite(f"./tmp/img_{img_count}.jpg", image_np_with_detections)


      count += 480
      img_count += 1
      vid.set(1, count)

      if cv2.waitKey(1) and 0xFF == ord('q'):
        print("Pressed Q")
        done = True

    else:
      vid.release()
      break
