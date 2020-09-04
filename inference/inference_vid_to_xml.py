import io
import re
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
import os
import pandas as pd
import glob
import xml.etree.ElementTree as ET

# Set memory growth to True so it won't crash
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Video and model path
model_path = 'exported-models/mobile_net/saved_model/'
video_path = 'data/vid_2/vid.mp4'
raw_path = 'data/vid_2/raw'

xml_format = r"""
<annotation verified="yes">
	<folder>raw</folder>
	<filename>{name}</filename>
	<path>{abs_path}</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>1280</width>
		<height>720</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	{objects}
</annotation>
"""

object_format = r"""
<object>
    <name>{name}</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
        <xmin>{xmin}</xmin>
        <ymin>{ymin}</ymin>
        <xmax>{xmax}</xmax>
        <ymax>{ymax}</ymax>
    </bndbox>
</object>
"""


def csv_to_xml(csv_input, output_path = ""):
    head, tail = os.path.split(csv_input)
    csv_name = tail
    df = pd.read_csv(csv_input)
    filenames = df.filename.unique()
    for filename in filenames:
        abs_path = os.path.abspath(csv_input).replace(csv_name, filename)
        yolo_path = os.path.join(output_path, filename.replace('jpg', 'xml'))
        df_tmp = df[df.filename == filename]
        objects = ""
        for row in df_tmp.iterrows():
            output = object_format.format(
                name = row[1]['class'],
                xmin = row[1]['xmin'],
                xmax = row[1]['xmax'],
                ymin = row[1]['ymin'],
                ymax = row[1]['ymax'])
            objects = objects + output
        output = xml_format.format(objects = objects, abs_path = abs_path, name = filename)
        with open(yolo_path, 'w') as f:
            f.write(output)



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


def detection_to_xml(detection, filename, threshold = 0.7):
    y = 720
    x = 1280
    abs_path = os.path.abspath(filename)
    # [y_min, x_min, y_max, x_max]
    category_index = {
        1: {'id': 1, 'name': 'player'},
        2: {'id': 2, 'name': 'trait'}
    }

    index_threshold = next(x[0] for x in enumerate(list(detection['detection_scores'].numpy()[0])) if x[1] < threshold)
    detection_classes = detection['detection_classes'][0].numpy().astype(np.int32)[:index_threshold]
    detection_boxes = detection['detection_boxes'][0].numpy()[:index_threshold]
    objects = ""

    for i in range(index_threshold):
        name = category_index.get(detection_classes[i]).get('name')
        ymin = int(detection_boxes[i][0] * y)
        xmin = int(detection_boxes[i][1] * x)
        ymax = int(detection_boxes[i][2] * y)
        xmax = int(detection_boxes[i][3] * x)
        obj = object_format.format(name = name, ymin = ymin, xmin = xmin, ymax = ymax, xmax = xmax)
        objects = objects + obj

    return xml_format.format(name = filename, abs_path = abs_path, objects = objects)


tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load(model_path)

vid = cv2.VideoCapture(video_path)

count = 0
img_count = 0
while vid.isOpened():
    done = False
    ret, frame = vid.read()
    if ret:
      # filename
      filename = f"img_{img_count}.jpg"
      file_path = os.path.join(raw_path, f"img_{img_count}.jpg") 
      print(file_path)
      xml_name = os.path.join(raw_path, f"img_{img_count}.xml")
      os.path.abspath(filename)
      input_tensor = np.expand_dims(frame, 0)
      detections = detect_fn(input_tensor)

      data = detection_to_xml(detections, filename, 0.7)
      cv2.imwrite(file_path, frame)

      with open(xml_name, 'w') as f:
          f.write(data)

      count += 60
      img_count += 1
      vid.set(1, count)

      if cv2.waitKey(1) and 0xFF == ord('q'):
        print("Pressed Q")
        done = True

    else:
      vid.release()
      break
