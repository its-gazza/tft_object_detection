# Object Detection for TFT

![](output.gif)

## Setup
Object detection module:
```
git clone https://github.com/tensorflow/models.git
cd models/research
C:\Users\Techfast\Documents\protoc-4.0.0-rc-1-win64\bin\protoc.exe object_detection/protos/*.proto --python_out=.
pip install cython
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

## Step 1: Get image and do image labelling
1. Create folder under `data` and create a subfolder called `raw` (e.g. ./data/vid_0/raw)
1. Download video using [download_video.py](./preprocess/video/download_video.py) 
```shell
python preprocess/video/download_video.py --url VIDEO_URL --output ./data/vid_0/vid.mp4
```
3. Generate image from downloaded video using [generate_image.py](./preprocess/video/generate_image.py)
```shell
# This will generate an image per 2 seconds
python preprocess/video/generate_image.py \
    --video ./data/vod_1/vid.mp4 \
    --output_dir ./data/vid_0/raw \
    --interval 2 \
    --showvideo
```
4. Use [labelImg](https://github.com/tzutalin/labelImg) to annotate images and saw annotations under the `raw` folder  

Note: you hsould annotatte in the PascalVOC format, it should return XML files

## Step 2: Create `record` files
1. Split the data:
```
python preprocess/tfrecord/partition_dataset.py -x -i data/vid_2/raw -o data/vid_2 -r 0.1
```


2. Generate record file
```
python preprocess/tfrecord/generate_tfrecord.py \
    -x data/vid_0/train \
    -l data/annotations/label_map.pbtxt \
    -o data/annotations/train_0.record

python preprocess/tfrecord/generate_tfrecord.py \
    -x data/vid_0/test \
    -l data/annotations/label_map.pbtxt \
    -o data/annotations/test_0.record
```

## Step 2: Configure pipeline file
1. Download a pre-trained-model to [pre-trained-models](./pre-trained-models), 
you can use any model you want from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
1. Update pipeline config file, you MUST update (search `PATH_TO_BE_CONFIGURED` in the config file):  
    1. input_path
    1. label_map_path
    1. num_classes
    1. fine_tune_checkpoint_type to `detection`
    1. reduce batch_size if you're running out of memories

## Step 3: Train model
Copy `model_train_tf2.py` and `exporter_main_V2.py` to root path
```shell
cp models/research/object_detection/model_main_tf2.py .
cp models/research/object_detection/exporter_main_v2.py .

python training/model_main_tf2.py --model_dir training/mobile_net --pipeline_config_path pre-trained-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config
```

## Step 4: Export model
```shell
python exporter_main_v2.py \
    --input_type image_tensor \
    --pipeline_config_path pre-trained-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config \
    --trained_checkpoint_dir training/mobile_net \
    --output_directory exported-models/mobile_net1
```

## Step 5: Inference model
1. Go to [inference_vid.py](./inference/inference_vid.py) and change `model_path` and `video_path`, note you MUST 
specify the `saved_model` as this is where the model is
1. Call the script
```shell
python ./inference/inference_vid.py
```