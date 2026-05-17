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

## Commercial support

For teams using this repo as a starting point for IP-camera ingest, RTSP reliability, YOLO inference, RTMP restreaming, Docker deployment, playback, or long-running edge-camera services, I offer a paid integration review:

- Review page: https://x2.brucelu.top/edgecam/?source=github-yolo-rtsp-object-detection-rtmp
- Sample deliverable: https://x2.brucelu.top/edgecam/sample/
- Checkout: https://x2.brucelu.top/edgecam/checkout/?source=github-yolo-rtsp-object-detection-rtmp
- Product catalog: https://x2.brucelu.top/products/?source=github-yolo-rtsp-object-detection-rtmp

Boundary: this is paid engineering review/support. It does not include camera credential handling, guaranteed model accuracy, managed surveillance operation, or production deployment ownership.
