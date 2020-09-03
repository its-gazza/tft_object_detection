# Object Detection for TFT

## Setup
Files:  
* Images and XML: `images/raw`
* label_map.pbtxt: under `annotations`

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

## Step 1: Create `record` files
1. Split the data:
```
python preprocess/tfrecord/partition_dataset.py -x -i data/vid_2/raw -o data/vid_2 -r 0.1
```


1. Generate record file
```
python preprocess/tfrecord/generate_tfrecord.py -x data/vid_0/train -l data/annotations/label_map.pbtxt -o data/annotations/train_0.record
python preprocess/tfrecord/generate_tfrecord.py -x data/vid_0/test -l data/annotations/label_map.pbtxt -o data/annotations/test_0.record

python preprocess/tfrecord/generate_tfrecord.py -x data/vid_1/train -l data/annotations/label_map.pbtxt -o data/annotations/train_1.record
python preprocess/tfrecord/generate_tfrecord.py -x data/vid_1/test -l data/annotations/label_map.pbtxt -o data/annotations/test_1.record

python preprocess/tfrecord/generate_tfrecord.py -x data/vid_2_/train -l data/annotations/label_map.pbtxt -o data/annotations/train_2_.record
python preprocess/tfrecord/generate_tfrecord.py -x data/vid_2_/test -l data/annotations/label_map.pbtxt -o data/annotations/test_2_.record
```

## Step 2: Configure pipeline file
1. Copy pre-trained model to `pre-trained-models`
1. Copy pre-trained model's pipeline.config to `model/ssd_resnet`
1. Update pipeline config file

## Step 3: Train model
Copy `model_train_tf2.py` and `exporter_main_V2.py` to root path
```shell
cp models/research/object_detection/model_main_tf2.py .
cp models/research/object_detection/exporter_main_v2.py .

python training/model_main_tf2.py --model_dir training/mobile_net --pipeline_config_path pre-trained-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config
```

## Step 4: Export model
```
python exporter_main_v2.py --input_type image_tensor --pipeline_config_path pre-trained-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config --trained_checkpoint_dir training/test --output_directory exported-models/mobile_net
```

## Appendix:
label_map.pbtxt:

```
item {
    id: 1
    name: 'cat'
}

item {
    id: 2
    name: 'dog'
}
```

114