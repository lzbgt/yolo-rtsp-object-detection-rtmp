from __future__ import division
import os
import sys
import time
import threading
import subprocess as sp
from collections import deque
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
import argparse
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

def write(x, img):
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

def arg_parse():
    """
    Parse arguements to the detect module
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.25)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "160", type = str)
    return parser.parse_args()


def open_cap():
    cap = cv2.VideoCapture("rtsp://admin:qwer1234@192.168.30.64:554/h264/ch1/sub/av_stream")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

image_queue = deque(maxlen=31)

def fn_enque_image():
    ret = False
    cap = None
    while True:
        if not ret:
            print('reopen rtsp')
            cap = open_cap()
        ret, frame = cap.read()
        if ret:
            image_queue.append(frame)
            #print('enqueued')

if __name__ == '__main__':
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    num_classes = 80

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()

    num_classes = 80
    bbox_attrs = 5 + num_classes
    
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    # -vcodec h264, -vcodec libx264 -acodec copy -pix_fmt yuv420p
    cmd = "/usr/local/bin/ffmpeg -framerate 4 -f image2pipe -vcodec mjpeg -i - -vcodec libx264 -vb 250k -framerate 4 -f flv 'rtmp://192.168.30.102/oflaDemo/ipc64 live=1'"
    proc = sp.Popen(cmd, stdin=sp.PIPE, shell=True)
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()
            
    model.eval()

    t = threading.Thread(target=fn_enque_image)
    t.start()
    
    frames = 0
    start = time.time()    
    classes = load_classes('data/coco.names')
    colors = pkl.load(open("pallete", "rb"))
    ret = True
    time.sleep(3)
    while True:
        try:
            frame = image_queue.popleft()
        except:
            print("Unexpected error:", sys.exc_info()[0])
            time.sleep(0.5)
            continue

        #print('image queue size:', len(image_queue))
        if frame is None:
            time.sleep(0.5)
            print('No Frame')
            continue
        img, orig_im, dim = prep_image(frame, inp_dim)                            
        if CUDA:
            img = img.cuda()
        
        output = model(Variable(img), CUDA)
        output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

        if type(output) == int:
            frames += 1
            _none, jpg = cv2.imencode('.jpg', orig_im)
            proc.stdin.write(jpg.tobytes())
            continue
        
        output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
        output[:,[1,3]] *= frame.shape[1]
        output[:,[2,4]] *= frame.shape[0]
        for x in output:
            write(x, orig_im)
        _none, jpg = cv2.imencode('.jpg', orig_im)
        proc.stdin.write(jpg.tobytes())
        frames += 1
        if frames % 180 == 0:
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
    

    
    

