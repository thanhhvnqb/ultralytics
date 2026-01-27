# VisDrone:
- Train: `yolo detect train data=VisDrone.yaml model=yolo26n-p2.yaml name=VisDrone/trainval/yolo26n-p2 imgsz=640 batch=32 nbs=128 epochs=200 device=0,1 pretrained=False`
    - Note that set `nbs=128` to accumodation gradient when need to set batch-size < 128.
- Validate: `yolo detect val data=VisDrone.yaml model=./runs/detect/VisDrone/trainval/yolo26n-ep200/weights/best.pt imgsz=640 batch=64 device=0`