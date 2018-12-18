# Automatic recycler

[Automatic recycler](https://i.imgur.com/v3OXxFc.png)

Raspberry Pi Automatic Recycler with Tensorflow object detection. 

# Introduction

To boost motivation of recycling we introduce a prototype of waste detector with automatic detection of recyclable items. 

Youtube video https://www.youtube.com/watch?v=Tpj1u7IZ9A0

# Software and Hardware requirements

## Hardware
- Raspberry Pi 3 B+ 
- Raspberry Camera Module V2
- 2 NEMA 17 stepper motors. Model: 17HS4401S 
- 3D printer to fix stepper motors together. For this project, the components were designed by Timothée Gerliner with Catia software and printed on Makerbot REPLICATOR+
- 2 Motor Drivers. Model: L298N
- 5V External power supply

Sample images for training mode were gathered from http://image-net.org/

For following categories:
- Beercan: n02823510
- Bottle: n02876657
- Box: n02883344
- Coffee can: n03062985
- Paper cup: n03216710
- Pen: n03906997
- Pop bottle: n03983396
- Soda Can: n04255586
- Paper: n06255613
- Apple: n07739125
- Banana: n07753592
- Egg: n07840804

## Software
Python v3.6.6
pip3 v18.0
Tensorflow v1.11

# Installation 

## Creating image database
At first, it is neccessary to gather enough image samples of categories that the garbage bin should recognize. ImageNet provides over 21000 categories with atleast hunders of images per each. 

To download images for research purpose a process can be automated with ImageNet downloadutils.py https://github.com/tzutalin/ImageNet_Utils

```
py downloadutils.py --downloadImages --wnid n02084071
```

## Training the model

To train the recognizion model, we use Tensorflow image retraining sample: https://github.com/tensorflow/hub/tree/master/examples/image_retraining 

To speed up training, the main computation of model can be done on host machine and then copied to Raspberry Pi.

- image_dir - directory that contains category images each separated into its own folder
- output_graph - directory where output graph will be saved. TensorFlow uses a dataflow graph to represent your computation in terms of the dependencies between individual operations.
- output_labels - directory where labels will be stored
- bottleneck_dir - directory where bottleneck will be stored. The bottleneck in a neural network is just a layer with less neurons then the layer below or above it. Having such a layer encourages the network to compress feature representations to best fit in the available space, in order to get the best loss during training.
- save_model_dir - directory where trained model will be saved

```
py retrain.py --image_dir=C:\Users\marek.ARR\Desktop\tensorflow\retrain\sample_training\flower_photos --output_graph=C:\Users\marek.ARR\Desktop\tensorflow\retrain\sample_training\graph\res1\output_graph.pb --output_labels=C:\Users\marek.ARR\Desktop\tensorflow\retrain\sample_training\labels\res1\output_labels.txt --bottleneck_dir=C:\Users\marek.ARR\Desktop\tensorflow\retrain\sample_training\bottleneck --saved_model_dir=C:\Users\marek.ARR\Desktop\tensorflow\retrain\sample_training\model\res1
```

## Testing a model

- graph - path to output graph file generated during training
- labels - path to labels file generated during training
- image - path to image to detect
- 

```
py label_image.py --graph=C:\Users\marek.ARR\Desktop\tensorflow\retrain\recycler_new_training\graph\res1\output_graph.pb --labels=C:\Users\marek.ARR\Desktop\tensorflow\retrain\recycler_new_training\labels\res1\output_labels.txt --image=C:\Users\marek.ARR\Desktop\tensorflow\retrain\to_classify\banana1.jpg
```

## Putting it all together

Launch recycler

```
python
```

### GPIO pins for Stepper 1

### GPIO pins for Stepper 2

# Further improvements

The ultimate goal of boosting circular economy would be to introduce machine with near-time object detection with incentive reward. Such machines are being deployed in Taiwan as i-Trash. http://www.itrash.com.tw

# Team

- Thomas Polus
- Marek Šíp
- Louis Yonatan
- 3D component design by Timothée Gerlinger
- Components and laboratory provided by KSU

2018, In cooperation with Mechatronics department of 경성대학교 / Kyungsung University in Busan, Republic of Korea.