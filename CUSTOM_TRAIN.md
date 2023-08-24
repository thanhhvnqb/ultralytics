<!---# Ultralytics YOLO 🚀, AGPL-3.0 license
# COCO 2017 dataset http://cocodataset.org by Microsoft
# Example usage: yolo train data=coco.yaml
# parent
# ├── ultralytics
# └── datasets
#     └── coco  ← downloads here (20.1 GB)-->
# Huấn luyện Yolov8 trên máy local

- Đoạn code trỏ đến file config (liên quan đến dataset_dir):
```
if os_name == 'Windows':
    path = Path.home() / 'AppData' / 'Roaming' / sub_dir
elif os_name == 'Darwin':  # macOS
    path = Path.home() / 'Library' / 'Application Support' / sub_dir
elif os_name == 'Linux':
    path = Path.home() / '.config' / sub_dir
else:
    raise ValueError(f'Unsupported operating system: {os_name}')
```
- Khi chạy huấn luyện, chạy lệnh ```./train.sh <ten_model>```. Cần cài đặt docker, docker compose, nvidia-docker để chạy được với GPU.
- Để test speed của model, chạy lệnh ```./speed.sh <ten_model>```.
- Link excel làm việc: https://docs.google.com/spreadsheets/d/1GpOcga7PgX1a2QyrQVaa36dww1n2GsXhqD7AUEivQwE/edit#gid=0

# Chuẩn bị cơ sở dữ liệu
- Tải file coco từ link này:
    - Cho detection: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip (Nên dùng chung với segmentation luôn cho khỏe)
    - Cho Segmentation: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip
    - Cho Pose: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-pose.zip
- Sau đó tải ảnh về và cho vào thư mục images

# Cài đặt docker, nvidia-docker và docker compose (đang bị lỗi exit khi kết thúc 1 epoch)
- Cài đặt docker: ```sudo apt install docker.io```
- Cài đặt Docker compose: theo hướng dẫn [tại đây](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-compose-on-ubuntu-22-04) (Lưu ý chọn version cho ubuntu)