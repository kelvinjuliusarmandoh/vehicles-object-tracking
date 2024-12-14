# Vehicles Object Tracking - YOLOv11

## Introduction
Object Tracking is a computer vision technique that allows machine to identify and follows motion of the object in a video or an image. In this project, we would like to track and identify of each object in a video / per frame. For the dataset, you can access the data [here](https://universe.roboflow.com/hertanto-weiyuan-fd3qo/cctv-8vtyi).

We would try using YOLOv11 from ultralytics and build a web app from streamlit for demo this project. 

If you want to run locally, you can run this command:

```streamlit run app.py```

![video.gif](./files/video-results.gif)
## Response
```
cls: tensor([0.], device='cuda:0')
conf: tensor([0.8254], device='cuda:0')
data: tensor([[515.5414, 207.8445, 615.1335, 285.1855,   0.8254,   0.0000]], device='cuda:0')
id: None
is_track: False
orig_shape: (480, 640)
shape: torch.Size([1, 6])
xywh: tensor([[565.3375, 246.5150,  99.5921,  77.3410]], device='cuda:0')
xywh: tensor([[565.3375, 246.5150,  99.5921,  77.3410]], device='cuda:0')
xywhn: tensor([[0.8833, 0.5136, 0.1556, 0.1611]], device='cuda:0')
xyxy: tensor([[515.5414, 207.8445, 615.1335, 285.1855]], device='cuda:0')
xyxyn: tensor([[0.8055, 0.4330, 0.9611, 0.5941]], device='cuda:0')
```

## Model Performance
!['results.png'](./files/results.png)
