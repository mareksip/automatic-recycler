
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import argparse

import sched
import time

import subprocess

import numpy as np
import tensorflow as tf

import RPi.GPIO as GPIO

import os.path

s = sched.scheduler(time.time, time.sleep)

file_name = "tensorflow/examples/label_image/data/grace_hopper.jpg"
model_file = \
    "tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb"
label_file = "tensorflow/examples/label_image/data/imagenet_slim_labels.txt"
input_height = 299
input_width = 299
input_mean = 0
input_std = 255
input_layer = "input"
output_layer = "InceptionV3/Predictions/Reshape_1"

threshold = 0.5

# define bin location
waste_bin = 90
biowaste_bin = 60
recycle_bin = -60

# last move needs to be remembered to get bin into initial position
last_move = 0

# step to register the last step that was made
laststep_file = "laststep.txt"

# optional argument to launch with certain position
init_location = 0


def load_graph(model_file):
    print ("Loading graph file...")
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    print ("Reading tensor from image...")
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(
        dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    print ("Loading labels...")
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def capture(sc):
    print ("Capturing...")

    cmd = "raspistill -hf -w 640 -h 480 -n -q 50 -o /home/pi/Desktop/sequence/capture.jpeg"
    subprocess.call(cmd, shell=True)

    print ("Clasifying...")

    graph = load_graph(model_file)
    t = read_tensor_from_image_file(
        file_name,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)

    # labels: apple, banana, egg, bottle, box, cup, paper, pen
    # Recyclable: bottle, box, cup, paper
    # Waste: pen and everything else
    # tincan, can, beercan
    # Biowaste: apple, banana

    for i in top_k:
        print(labels[i], results[i])
        if results[i] > np.float32(threshold):
            print("bingo!")
            if (labels[i] == 'bottle'):
                print(labels[i])
                mover(1, 100)
                time.sleep(1.5)
                mover(2, -200)
                time.sleep(1.5)
                mover(1, -100)
                s.enter(5, 1, capture, (sc,))
                break
            if (labels[i] == 'cup') or (labels[i] == 'paper') or (labels[i] == 'box'):
                print(labels[i])
                mover(1, -100)
                time.sleep(0.2)
                mover(2, -200)
                time.sleep(0.2)
                mover(1, 100)
                s.enter(5, 1, capture, (sc,))
                break
            if (labels[i] == 'tincan') or (labels[i] == 'can') or (labels[i] == 'beercan'):
                print(labels[i])
                mover(1, 100)
                time.sleep(0.2)
                mover(2, -200)
                time.sleep(0.2)
                mover(1, -100)
                s.enter(5, 1, capture, (sc,))
                break
            if (labels[i] == 'banana') or (labels[i] == 'apple') or (labels[i] == 'egg'):
                print(labels[i])
                mover(1, 200)
                time.sleep(1.5)
                mover(2, -200)
                time.sleep(1.5)
                mover(1, -200)
                s.enter(5, 1, capture, (sc,))
                break
            else:
                print("other")
                mover(2, -200)
                s.enter(5, 1, capture, (sc,))
                break
        else:
            print("other")
            mover(2, -200)
            s.enter(5, 1, capture, (sc,))
            break


def mover(stepper, value):

    # log last move
    # if stepper == 1:
    #     write_last(value)

    # Pins for stepper 1
    out1 = 13
    out2 = 11
    out3 = 15
    out4 = 12

    if stepper == 2:
        out1 = 37
        out2 = 36
        out3 = 40
        out4 = 35

    i = 0
    positive = 0
    negative = 0
    y = 0

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(out1, GPIO.OUT)
    GPIO.setup(out2, GPIO.OUT)
    GPIO.setup(out3, GPIO.OUT)
    GPIO.setup(out4, GPIO.OUT)

    GPIO.output(out1, GPIO.LOW)
    GPIO.output(out2, GPIO.LOW)
    GPIO.output(out3, GPIO.LOW)
    GPIO.output(out4, GPIO.LOW)
    x = value

    if x > 0 and x <= 400:
        for y in range(x, 0, -1):
            if negative == 1:
                if i == 7:
                    i = 0
                else:
                    i = i+1
                y = y+2
                negative = 0
            positive = 1
            # print((x+1)-y)
            if i == 0:
                GPIO.output(out1, GPIO.HIGH)
                GPIO.output(out2, GPIO.LOW)
                GPIO.output(out3, GPIO.LOW)
                GPIO.output(out4, GPIO.LOW)
                time.sleep(0.01)
                # time.sleep(1)
            elif i == 1:
                GPIO.output(out1, GPIO.HIGH)
                GPIO.output(out2, GPIO.HIGH)
                GPIO.output(out3, GPIO.LOW)
                GPIO.output(out4, GPIO.LOW)
                time.sleep(0.01)
                # time.sleep(1)
            elif i == 2:
                GPIO.output(out1, GPIO.LOW)
                GPIO.output(out2, GPIO.HIGH)
                GPIO.output(out3, GPIO.LOW)
                GPIO.output(out4, GPIO.LOW)
                time.sleep(0.01)
                # time.sleep(1)
            elif i == 3:
                GPIO.output(out1, GPIO.LOW)
                GPIO.output(out2, GPIO.HIGH)
                GPIO.output(out3, GPIO.HIGH)
                GPIO.output(out4, GPIO.LOW)
                time.sleep(0.01)
                # time.sleep(1)
            elif i == 4:
                GPIO.output(out1, GPIO.LOW)
                GPIO.output(out2, GPIO.LOW)
                GPIO.output(out3, GPIO.HIGH)
                GPIO.output(out4, GPIO.LOW)
                time.sleep(0.01)
                # time.sleep(1)
            elif i == 5:
                GPIO.output(out1, GPIO.LOW)
                GPIO.output(out2, GPIO.LOW)
                GPIO.output(out3, GPIO.HIGH)
                GPIO.output(out4, GPIO.HIGH)
                time.sleep(0.01)
                # time.sleep(1)
            elif i == 6:
                GPIO.output(out1, GPIO.LOW)
                GPIO.output(out2, GPIO.LOW)
                GPIO.output(out3, GPIO.LOW)
                GPIO.output(out4, GPIO.HIGH)
                time.sleep(0.01)
                # time.sleep(1)
            elif i == 7:
                GPIO.output(out1, GPIO.HIGH)
                GPIO.output(out2, GPIO.LOW)
                GPIO.output(out3, GPIO.LOW)
                GPIO.output(out4, GPIO.HIGH)
                time.sleep(0.01)
                # time.sleep(1)
            if i == 7:
                i = 0
                continue
            i = i+1

    elif x < 0 and x >= -400:
        x = x*-1
        for y in range(x, 0, -1):
            if positive == 1:
                if i == 0:
                    i = 7
                else:
                    i = i-1
                y = y+3
                positive = 0
            negative = 1
            # print((x+1)-y)
            if i == 0:
                GPIO.output(out1, GPIO.HIGH)
                GPIO.output(out2, GPIO.LOW)
                GPIO.output(out3, GPIO.LOW)
                GPIO.output(out4, GPIO.LOW)
                time.sleep(0.01)
                # time.sleep(1)
            elif i == 1:
                GPIO.output(out1, GPIO.HIGH)
                GPIO.output(out2, GPIO.HIGH)
                GPIO.output(out3, GPIO.LOW)
                GPIO.output(out4, GPIO.LOW)
                time.sleep(0.01)
                # time.sleep(1)
            elif i == 2:
                GPIO.output(out1, GPIO.LOW)
                GPIO.output(out2, GPIO.HIGH)
                GPIO.output(out3, GPIO.LOW)
                GPIO.output(out4, GPIO.LOW)
                time.sleep(0.01)
                # time.sleep(1)
            elif i == 3:
                GPIO.output(out1, GPIO.LOW)
                GPIO.output(out2, GPIO.HIGH)
                GPIO.output(out3, GPIO.HIGH)
                GPIO.output(out4, GPIO.LOW)
                time.sleep(0.01)
                # time.sleep(1)
            elif i == 4:
                GPIO.output(out1, GPIO.LOW)
                GPIO.output(out2, GPIO.LOW)
                GPIO.output(out3, GPIO.HIGH)
                GPIO.output(out4, GPIO.LOW)
                time.sleep(0.01)
                # time.sleep(1)
            elif i == 5:
                GPIO.output(out1, GPIO.LOW)
                GPIO.output(out2, GPIO.LOW)
                GPIO.output(out3, GPIO.HIGH)
                GPIO.output(out4, GPIO.HIGH)
                time.sleep(0.01)
                # time.sleep(1)
            elif i == 6:
                GPIO.output(out1, GPIO.LOW)
                GPIO.output(out2, GPIO.LOW)
                GPIO.output(out3, GPIO.LOW)
                GPIO.output(out4, GPIO.HIGH)
                time.sleep(0.01)
                # time.sleep(1)
            elif i == 7:
                GPIO.output(out1, GPIO.HIGH)
                GPIO.output(out2, GPIO.LOW)
                GPIO.output(out3, GPIO.LOW)
                GPIO.output(out4, GPIO.HIGH)
                time.sleep(0.01)
                # time.sleep(1)
            if i == 0:
                i = 7
                continue
            i = i-1
    GPIO.cleanup()


# Reads text line from last_step.txt
# Text line contains int of last move
def read_last():
    print('reading last..')

    ress = os.path.isfile(laststep_file)

    if ress == False:
        # no need to read return 0
        # shall be in initial position
        return 0
    elif ress == True:
        # return last position
        file = open(laststep_file, 'r')
        return file.read()

# Writes text line to last_step.txt
# Text line containst int of last move


def write_last(value):
    print('writing last..')
    file = open(laststep_file, 'w')
    file.write(str(value))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed")
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    parser.add_argument("--init_position", type=int,
                        help="initial position of stepper motor")
    parser.add_argument("--waste_bin", type=int, help="waste bin stepper move")
    parser.add_argument("--biowaste_bin", type=int,
                        help="biowaste bin stepper moves")
    parser.add_argument("--recycle_bin", type=int,
                        help="recycle bin stepper moves")
    args = parser.parse_args()

    if args.graph:
        model_file = args.graph
    if args.image:
        file_name = args.image
    if args.labels:
        label_file = args.labels
    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer
    if args.init_position:
        init_location = args.init_position
    if args.waste_bin:
        waste_bin = args.waste_bin
    if args.biowaste_bin:
        biowaste_bin = args.biowaste_bin
    if args.recycle_bin:
        recycle_bin = args.recycle_bin

    # Check for init position
    if init_location != 0:
        print ('init location: ', str(init_location))
        mover(1, init_location)

    lastmove = read_last()
    print ('last move: ', str(lastmove))
    # if(lastmove != 0):
    #    mover(1, int(lastmove) * 1)
        
    # print bin locations
    print ('waste bin: ', str(waste_bin), ' biowaste bin: ',
           str(biowaste_bin), ' recycle bin: ', str(recycle_bin))

    # write_last(50)
    # fileval = read_last()
    # print (fileval)
    print ("Launching timer..")
    s.enter(1, 1, capture, (s,))
    s.run()
