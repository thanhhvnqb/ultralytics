<!---# Ultralytics YOLO 🚀, AGPL-3.0 license
# COCO 2017 dataset http://cocodataset.org by Microsoft
# Example usage: yolo train data=coco.yaml
# parent
# ├── ultralytics
# └── datasets
#     └── coco  ← downloads here (20.1 GB)-->
# Huấn luyện Yolov8 trên máy local

- Khi chạy huấn luyện, chạy lệnh ```docker compose up```. Cần cài đặt docker, docker compose, nvidia-docker để chạy được với GPU.
- Link excel làm việc: https://docs.google.com/spreadsheets/d/1GpOcga7PgX1a2QyrQVaa36dww1n2GsXhqD7AUEivQwE/edit#gid=0

# Chuẩn bị cơ sở dữ liệu
- Tải file coco từ link này:
    - Cho detection: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip
    - Cho Segmentation: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip
    - Cho Pose: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-pose.zip
- Sau đó tải ảnh về và cho vào thư mục images

# Cài đặt docker, nvidia-docker và docker compose
- Cài đặt docker: ```sudo apt install docker.io```
- Cài đặt Docker compose: theo hướng dẫn [tại đây](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-compose-on-ubuntu-22-04) (Lưu ý chọn version cho ubuntu)