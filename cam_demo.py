from __future__ import division
import os
import sys, traceback
import time
import threading
import subprocess as sp
from collections import deque
import click
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import inp_to_image
import pandas as pd
import random 
import pickle as pkl

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img, classes, colors):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img

def open_cap(vuri, ifps, buffsize):
    cap = cv2.VideoCapture(vuri)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffsize)
    cap.set(cv2.CAP_PROP_FPS, ifps)
    return cap

image_queue = None
image_write_queue = None

def fn_enque_image(rtsp_url, ifps, buffsize):
    ret = False
    cap = None
    while True:
        if not ret:
            print('reopen rtsp')
            cap = open_cap(rtsp_url, ifps, buffsize)
        ret, frame = cap.read()
        if ret:
            try:
                image_queue.append(frame)
            except:
                print("Unexpected error:", sys.exc_info()[0])

def fn_write_image(proc):
    while True:
        try:
            image = image_write_queue.popleft()
            _none, jpg = cv2.imencode('.jpg', image)
            proc.stdin.write(jpg.tobytes())
        except:
            # print(traceback.format_exc())
            time.sleep(0.01)


@click.command()
@click.option("--rtsp", default="rtsp://admin:qwer1234@192.168.30.64:554/h264/ch1/sub/av_stream", help="rtsp url of ipcamera")
@click.option("--ifps", default=25, help="fps of rtsp stream, eg. 25")
@click.option("--rtmp", default="rtmp://localhost/oflaDemo/ipc64 live=1", help="url of rtmp server")
@click.option("--ofps", default=3, help="output rtmp fps, eg. 4")
@click.option("--weights", default="yolov3.weights", help="path to yolo weights")
@click.option("--size", default="", help="output video size. eg. 680x460")
@click.option("--buffsize", default=4, help="queque size for rtsp")
@click.option("--confidence", default = 0.25, help = "Object Confidence to filter predictions")
@click.option("--nms_thresh", default = 0.4, help = "NMS Threshhold")
@click.option("--reso", default = "160", help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed")      
def livestream(rtsp, ifps, rtmp, ofps, weights, size, buffsize, confidence, nms_thresh,reso):
    global image_queue
    global image_write_queue

    image_queue = deque(maxlen=buffsize)
    image_write_queue = deque(maxlen=ifps)

    cfgfile = "cfg/yolov3.cfg"
    weightsfile = weights

    num_classes = 80
    CUDA = torch.cuda.is_available()
    
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    
    model.net_info["height"] = reso
    inp_dim = int(model.net_info["height"])
    # -vcodec h264, -vcodec libx264 -acodec copy -pix_fmt yuv420p
    #cmd = "ffmpeg -framerate 4 -f image2pipe -vcodec mjpeg -i - -vcodec libx264 -vb 250k -framerate 4 -g 24 -f flv 'rtmp://localhost/oflaDemo/ipc64 live=1'"
    if size:
        size = "-s " + size
    
    cmd = "ffmpeg -framerate {ofps} -f image2pipe -vcodec mjpeg -i - -vcodec libx264 {size} -vb 250k -framerate {ofps} -g 24 -f flv '{rtmp}'".format(ofps=ofps, rtmp=rtmp, size=size)
    print('cmd: ', cmd)
    proc = sp.Popen(cmd, stdin=sp.PIPE, shell=True)
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model.eval()

    t = threading.Thread(target=fn_enque_image, args=(rtsp, ifps, buffsize))
    t.start()

    t2 = threading.Thread(target=fn_write_image, args=(proc,))
    t2.start()
    
    frames = 0
    start = time.time()
    classes = load_classes('data/coco.names')
    colors = pkl.load(open("pallete", "rb"))
    time.sleep(3)
    raw = False
    start = 0

    while True:
        try:
            frame = image_queue.popleft()
        except:
            print("Unexpected error:", sys.exc_info()[0])
            time.sleep(0.01)
            continue

        if raw:
            _none, jpg = cv2.imencode('.jpg', frame)
            proc.stdin.write(jpg.tobytes(), classes, colors)
            frames += 1
            if frames % 1800 == 0:
               print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            continue

        if frame is None:
            time.sleep(0.5)
            print('No Frame')
            continue
        img, orig_im, dim = prep_image(frame, inp_dim)
        if CUDA:
            img = img.cuda()
        
        output = model(Variable(img), CUDA)
        output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thresh)

        if type(output) == int:
            frames += 1
            image_write_queue.append(orig_im)
            continue
        
        output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
        output[:,[1,3]] *= frame.shape[1]
        output[:,[2,4]] *= frame.shape[0]
        for x in output:
            write(x, orig_im, classes, colors)

        image_write_queue.append(orig_im)
        frames += 1
        if frames % 1800 == 0:
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

if __name__ == '__main__':
    livestream()
    

    
    

