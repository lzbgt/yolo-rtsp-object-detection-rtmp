# A PyTorch implementation of a YOLO v3 Object Detector, RTSP to RTMP specific

```
pip install -r requirements.txt
pythonw rtsp_proc_rtmp.py --help
Usage: rtsp_proc_rtmp.py [OPTIONS]

Options:
  --rtsp TEXT         rtsp url of ipcamera
  --ifps INTEGER      fps of rtsp stream, eg. 25
  --rtmp TEXT         url of rtmp server
  --ofps INTEGER      output rtmp fps, eg. 4
  --weights TEXT      path to yolo weights
  --size TEXT         output video size. eg. 680x460
  --buffsize INTEGER  queque size for rtsp
  --confidence FLOAT  Object Confidence to filter predictions
  --nms_thresh FLOAT  NMS Threshhold
  --reso TEXT         Input resolution of the network. Increase to increase
                      accuracy. Decrease to increase speed
  --help              Show this message and exit.
```