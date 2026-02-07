# Hướng dẫn: Cách YOLO gán Object cho từng Detection Head

## 1. Tổng quan kiến trúc Detection Head

### 1.1. Detection Head là gì?
Trong YOLO, **Detection Head** là các lớp đầu ra cuối cùng của mô hình, chịu trách nhiệm dự đoán:
- **Bounding boxes** (tọa độ vị trí object)
- **Class probabilities** (xác suất các class)

YOLO sử dụng **Multi-scale Detection** với nhiều detection heads tương ứng với các scales khác nhau.

#### Ví dụ cấu hình phổ biến:

**Cấu hình 3 scales (Standard YOLO):**
- **Head 1** (stride=8, P3): Phát hiện object nhỏ
- **Head 2** (stride=16, P4): Phát hiện object trung bình  
- **Head 3** (stride=32, P5): Phát hiện object lớn

**Cấu hình 2 scales (Your Config - yoloav1.yaml):**
- **Head 1** (stride=4, P2): Phát hiện object rất nhỏ (extra small)
- **Head 2** (stride=8, P3): Phát hiện object nhỏ (small)

> 💡 **Lưu ý:** Số lượng heads được **tự động phát hiện** từ config YAML qua `len(ch)` - không cần cấu hình cứng!

### 1.2. Vị trí trong code
File chính: [`ultralytics/nn/modules/head.py`](ultralytics/nn/modules/head.py)

```python
class Detect(nn.Module):
    """YOLO Detect head for object detection models."""
    
    def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = ()):
        self.nc = nc  # số classes
        self.nl = len(ch)  # số detection layers - TỰ ĐỘNG từ config
        self.reg_max = reg_max  # DFL channels
        self.no = nc + self.reg_max * 4  # outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides - tính động trong quá trình build
        
        # Box regression heads - số lượng heads = len(ch)
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) 
            for x in ch  # ch từ YAML: VD [128, 256] cho 2 heads hoặc [256, 512, 1024] cho 3 heads
        )
        
        # Classification heads - tương tự
        self.cv3 = nn.ModuleList(...)
```

#### Cách stride được tính tự động:
```python
# File: ultralytics/nn/tasks.py, line ~410
def __init__(self, cfg='yolo26.yaml', ch=3, nc=None, verbose=True):
    ...
    # Forward pass với ảnh dummy để tính stride
    m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])
    # VD: input 640x640
    # - Output shape 160x160 → stride = 640/160 = 4
    # - Output shape 80x80   → stride = 640/80  = 8
    # - Output shape 40x40   → stride = 640/40  = 16
```

## 2. Cơ chế gán Object cho Detection Head

### 2.1. Task-Aligned Assigner (TAA)
File: [`ultralytics/utils/tal.py`](ultralytics/utils/tal.py)

**TaskAlignedAssigner** là thuật toán chính để gán ground truth objects cho các anchor points (detection heads).

```python
class TaskAlignedAssigner(nn.Module):
    """
    Gán ground-truth objects cho anchors dựa trên task-aligned metric
    
    Tham số:
    - topk (int): Số lượng top candidates xem xét (mặc định 13)
    - num_classes (int): Số classes
    - alpha (float): Trọng số cho classification (mặc định 1.0)
    - beta (float): Trọng số cho localization (mặc định 6.0)
    - stride (list): [8, 16, 32] - stride của các detection heads
    """
```

### 2.2. Quy trình gán Object

#### Bước 1: Tạo Anchor Points
```python
# File: ultralytics/utils/tal.py -> make_anchors()
anchor_points, stride_tensor = make_anchors(feats, stride, grid_cell_offset=0.5)
# stride được lấy từ m.stride - tự động detect từ model
```

Mỗi pixel trên feature map tương ứng với 1 anchor point.

**Ví dụ Standard YOLO (3 scales, stride=[8, 16, 32]):**
- Feature map 80×80 (stride=8) → 6400 anchor points
- Feature map 40×40 (stride=16) → 1600 anchor points
- Feature map 20×20 (stride=32) → 400 anchor points
- **Tổng: 8400 anchor points**

**Ví dụ Your Config (2 scales, stride=[4, 8]):**
- Feature map 160×160 (stride=4) → **25,600 anchor points**
- Feature map 80×80 (stride=8) → **6,400 anchor points**
- **Tổng: 32,000 anchor points** ⚡ (nhiều hơn 3.8 lần!)

#### Bước 2: Lọc Candidates trong GT Box
```python
def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt):
    """
    Chọn các anchor points nằm TRONG ground truth boxes
    
    Returns:
        mask_in_gts: (b, n_boxes, h*w) - mask cho anchors nằm trong GT
    """
    # Tính khoảng cách từ anchor đến các cạnh của GT box
    lt, rb = gt_bboxes.chunk(2, 2)  # left-top, right-bottom
    bbox_deltas = torch.cat((xy_centers - lt, rb - xy_centers), dim=2)
    # Nếu tất cả deltas > 0 → anchor nằm trong box
    return bbox_deltas.amin(3).gt_(0)
```

#### Bước 3: Tính Alignment Metric
```python
def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
    """
    Tính độ phù hợp giữa prediction và GT
    
    Công thức: alignment = (classification_score^alpha) * (IoU^beta)
    """
    # Lấy classification score của class tương ứng
    bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]
    
    # Tính IoU giữa predicted box và GT box
    overlaps[mask_gt] = bbox_iou(gt_boxes, pd_boxes, CIoU=True)
    
    # Alignment metric kết hợp cả 2 yếu tố
    align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
    return align_metric, overlaps
```

**Ý nghĩa:**
- `alpha=1.0`: Classification score ảnh hưởng tuyến tính
- `beta=6.0`: IoU ảnh hưởng mạnh hơn (IoU^6)
- → Ưu tiên anchors có vị trí tốt (IoU cao) hơn là classification score

#### Bước 4: Select Top-K Candidates
```python
def select_topk_candidates(self, metrics, topk=13):
    """
    Với mỗi GT object, chọn top-k anchors có alignment metric cao nhất
    
    Args:
        metrics: (b, n_boxes, h*w) - alignment scores
        topk: số lượng top candidates (mặc định 13)
    
    Returns:
        mask_topk: (b, n_boxes, h*w) - mask cho top-k anchors
    """
    topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
    # Tạo mask cho các top-k positions
    count_tensor.scatter_add_(-1, topk_idxs, ones)
```

**Ví dụ minh họa:**
```
GT Object: Con chó ở tọa độ (100, 150, 200, 250)

Alignment scores của tất cả 8400 anchors:
- Anchor 1234 (scale 8x8):   0.85
- Anchor 2341 (scale 8x8):   0.82
- Anchor 1555 (scale 16x16): 0.79
- Anchor 7234 (scale 32x32): 0.15  ← Too far, low IoU
...

→ Chọn top-13 anchors có score cao nhất
→ Những anchors này sẽ được gán cho object "chó"
```

#### Bước 5: Xử lý Conflict (1 anchor → nhiều GT)
```python
def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
    """
    Nếu 1 anchor được gán cho nhiều GT objects → chọn GT có IoU cao nhất
    """
    fg_mask = mask_pos.sum(-2)  # Đếm số GT mỗi anchor được gán
    
    if fg_mask.max() > 1:  # Có conflict
        # Chọn GT có overlap cao nhất
        max_overlaps_idx = overlaps.argmax(1)
        mask_pos = is_max_overlaps  # Chỉ giữ lại 1 GT
```

#### Bước 6: Tạo Targets
```python
def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
    """
    Tạo target labels, bboxes, scores cho các positive anchors
    
    Returns:
        target_labels: (b, h*w) - class labels
        target_bboxes: (b, h*w, 4) - bbox coordinates  
        target_scores: (b, h*w, num_classes) - one-hot encoded scores
    """
    # Lấy GT tương ứng cho mỗi anchor
    target_labels = gt_labels.flatten()[target_gt_idx]
    target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]
    
    # Tạo one-hot encoding cho classes
    target_scores = torch.zeros((bs, h*w, num_classes))
    target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
    
    # Chỉ giữ targets cho foreground anchors
    target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)
```
m = model.model[-1]  # Detect() module
        self.stride = m.stride  # LẤY TỰ ĐỘNG từ model (có thể là [4,8], [8,16,32], v.v.)
        
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist()  # ĐỘNG - tự adapt với config
class v8DetectionLoss:
    def __init__(self, model, tal_topk=10):
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=[8, 16, 32]
        )
        self.bbox_loss = BboxLoss(reg_max)
        self.bce = nn.BCEWithLogitsLoss()
    
    def get_assigned_targets_and_loss(self, preds, batch):
        """Tính loss dựa trên target assignment"""
        
        # 1. Decode predictions
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        
        # 2. Gán targets sử dụng TaskAlignedAssigner
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),      # Classification scores
            pred_bboxes.detach(),                # Predicted boxes
            anchor_points * stride_tensor,       # Anchor positions
            gt_labels,                           # Ground truth labels
            gt_bboxes,                           # Ground truth boxes
            mask_gt                              # Valid GT mask
        )
        
        # 3. Tính losses
        # Classification loss - tất cả 8400 anchors
        loss_cls = self.bce(pred_scores, target_scores).sum() / target_scores_sum
        
        # Bbox loss - chỉ positive anchors (fg_mask=True)
        if fg_mask.sum():
            loss_box, loss_dfl = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                fg_mask
            )
```
×640 với 2 objects:
- Object 1: Person tại (100, 100, 300, 400) - Large object (200×300 pixels)
- Object 2: Cup tại (450, 450, 480, 480) - Small object (30×30 pixels)

### Kết quả Assignment với Standard YOLO (3 scales):

```
Feature Map Scale 8×8 (80×80 grid, 6400 anchors):
├─ Object 2 (Cup - small, 30×30px): 
│  └─ Top-13 anchors gần vị trí (56, 58) được gán
│     Anchors: [4503, 4504, 4583, 4584, 4585, ...]
│     
Feature Map Scale 16×16 (40×40 grid, 1600 anchors):
├─ Object 1 (Person - medium, 200×300px): 
│  └─ Top-13 anchors gần vị trí (13, 18) được gán
│     Anchors: [6933, 6934, 6973, 6974, ...]
│     
Feature Map Scale 32×32 (20×20 grid, 400 anchors):
├─ Object 1 (Person - large):
│  └─ Top-13 anchors gần center được gán
│     Anchors: [8234, 8235, 8254, ...]

Tổng: ~39 positive anchors (26 cho person, 13 cho cup)
      ~8361 negative anchors
```

### Kết quả Assignment với Your Config (2 scales, stride=[4, 8]):

```
Feature Map Scale 4×4 (160×160 grid, 25,600 anchors):
├─ Object 2 (Cup - small, 30×30px): 
│  └─ Top-13 anchors gần vị trí (112, 115) được gán
│     Anchors: [17,903, 17,904, 18,063, 18,064, ...]
│     IoU cao hơn vì resolution cao hơn!
│     
├─ Object 1 (Person - medium, 200×300px):
│  └─ Top-13 anchors cũng được gán ở scale này
│     Anchors: [4,000-4,500 range...]
│     
Feature Map Scale 8×8 (80×80 grid, 6400 anchors):
├─ Object 1 (Person): 
│  └─ Top-13 anchors bổ sung
│     Anchors: [3,000-3,500 range...]

Tổng: ~39 positive anchors (nhiều hơn ở scale nhỏ)
      ~31,961 negative anchors

✅ Ưu điểm: Phát hiện tốt objects nhỏ hơn (stride=4)
⚠️ Nhược điểm: Nhiều anchors hơn → tính toán chậm hơn8254, ...]

Tổng: ~39 positive anchors (26 cho person, 13 cho cup)
      ~8361 negative anchors
```

## 5. Các hàm quan trọng

### 5.1. make_anchors()
```python
# File: ultralytics/utils/tal.py
def make_anchors(feats, strides, grid_cell_offset=0.5):
    """
    Tạo anchor points từ feature maps
    
    Args:
        feats: List feature maps [(b,c,80,80), (b,c,40,40), (b,c,20,20)]
        strides: [8, 16, 32]
        grid_cell_offset: 0.5 (center của mỗi grid cell)
    
    Returns:
        anchor_points: (8400, 2) - [x, y] coordinates
        stride_tensor: (8400, 1) - stride cho mỗi anchor
    """
```

### 5.2. dist2bbox()
```python
# File: ultralytics/utils/tal.py
def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """
    Chuyển đổi distance predictions thành bbox coordinates
    
    Args:
        distance: (b, 8400, 4) - [left, top, right, bottom] distances
        anchor_points: (8400, 2) - anchor centers
    
    Returns:
        bboxes: (b, 8400, 4) - [x1, y1, x2, y2] hoặc [x, y, w, h]
    """
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return torch.cat([x1y1, x2y2], dim)  # xyxy format
```

### 5.3. bbox_iou()
```python
# File: ultralytics/utils/metrics.py
def bbox_iou(box1, box2, xywh=False, GIoU=False, DIoU=False, CIoU=True):
    """
    Tính IoU giữa predicted và ground truth boxes
    
    Args:
        box1: (n, 4) predicted boxes
        box2: (n, 4) ground truth boxes
        CIoU: Complete IoU (tính cả distance và aspect ratio)
    
    Retur`: **TỰ ĐỘNG** từ config YAML
  - Standard: `[8, 16, 32]`
  - Your config: `[4, 8]`
  - Có thể custom: `[4, 8, 16, 32, 64]` cho 5 scales!
        iou: (n, 1) IoU scores
    """
```

## 6. Tóm tắt

### Quy trình chính:
1. **Backbone** → 3 feature maps (P3, P4, P5)
2. **Detection Heads** → Predictions cho 8400 anchors
3. **TaskAlignedAssigner**:
   - Lọc anchors trong GT boxes
   - Tính alignment metric (classification × IoU)
   - Chọn top-k anchors cho mỗi object
   - Xử lý conflicts
   - Tạo target labels/boxes/scores
4. **Loss Computation**:
   - Classification loss: All anchors
   - Bbox loss: Only positive anchors
   - DFL loss: Only positive anchors

### Các tham số quan trọng:
- `topk=13`: Số anchors được gán cho mỗi object
- `alpha=0.5`: Trọng số classification trong alignment
- `beta=6.0`: Trọng số localization trong alignment
- `stride=[8, 16, 32]`: Multi-scale detection

### Ưu điểm của Task-Aligned Assignment:
✅ Dynamic assignment (không cố định anchor)
✅ Kết hợp cả classification và localization
✅ Xử lý tốt objects ở nhiều scales
✅ Giảm conflicts giữa các objects
### 7.1. Ví dụ Standard YOLO (3 scales)

```python
import torch
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors

# Giả lập predictions - 3 scales
feats = [
    torch.randn(1, 64, 80, 80),   # P3 - stride 8
    torch.randn(1, 128, 40, 40),  # P4 - stride 16
    torch.randn(1, 256, 20, 20),  # P5 - stride 32
]
strides = torch.tensor([8, 16, 32])

# Tạo anchors
anchor_points, stride_tensor = make_anchors(feats, strides, 0.5)
print(f"Total anchors: {anchor_points.shape[0]}")  # 8400
print(f"Strides: {strides.tolist()}")

# Giả lập GT: 1 object - person
gt_labels = torch.tensor([[[0]]])  # class 0
gt_bboxes = torch.tensor([[[100, 100, 300, 400]]])  # xyxy
mask_gt = torch.ones(1, 1, 1)

# Predictions
pred_scores = torch.randn(1, 8400, 80).sigmoid()
pred_bboxes = torch.randn(1, 8400, 4)

# Assignment
assigner = TaskAlignedAssigner(topk=13, num_classes=80, stride=strides.tolist())
_, target_bboxes, target_scores, fg_mask, target_gt_idx = assigner(
    pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt
)

# Kết quả
print(f"Positive anchors: {fg_mask.sum().item()}")  # ~13
print(f"Negative anchors: {(~fg_mask).sum().item()}")  # ~8387

# Xem anchors nào được gán
positive_anchor_indices = torch.where(fg_mask[0])[0]
print(f"Assigned anchor indices: {positive_anchor_indices}")
```
## 8. Câu hỏi Thường gặp (FAQ)

### Q1: Config của tôi chỉ có 2 scales (P2, P3) thay vì 3 scales. Có vấn đề gì không?
**A:** Không có vấn đề! Code tự động adapt:
- `self.nl = len(ch)` → tự detect số heads từ YAML
- `stride` được tính động qua forward pass
- TaskAlignedAssigner nhận `stride.tolist()` động

### Q2: Tại sao config tôi dùng stride=[4, 8] nhỏ hơn standard [8, 16, 32]?
**A:** Ưu điểm:
- ✅ Phát hiện objects **rất nhỏ** tốt hơn (stride=4 → resolution cao)
- ✅ Phù hợp cho datasets có nhiều small objects

Nhược điểm:
- ⚠️ Nhiều anchors hơn (32K vs 8.4K) → **chậm hơn ~4×**
- ⚠️ Tốn nhiều memory hơn
- ⚠️ Training có thể unstable nếu batch size nhỏ

### Q3: Làm sao thêm thêm detection heads (VD: 4 hoặc 5 scales)?
**A:** Chỉnh YAML config:
```yaml
head:
  # Thêm các upsampling/downsampling layers
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # P1
  - [..., 1, Conv, [256, 3, 2]]  # P6
  
  # Cuối cùng list tất cả heads
  - [[P1, P2, P3, P4], 1, Detect, [nc]]  # 4 heads
```
→ Code tự động adapt, không cần sửa gì thêm!

### Q4: topk=13 có phù hợp với 2 scales không?
**A:** Có thể cần điều chỉnh:
- Standard (3 scales): `topk=10-13` là phù hợp
- Your config (2 scales, nhiều anchors): có thể tăng `topk=15-20`
- Ít scales (1-2): có thể cần `topk` lớn hơn để compensate

Thử nghiệm:
```python
# File: train script
model = YOLO('yoloav1.yaml')
model.train(
    data='coco.yaml',
    tal_topk=15,  # Thử các giá trị khác nhau
    ...
)
```

### Q5: Với stride=[4, 8], objects lớn có bị detect kém không?
**A:** Có thể! Scale nhỏ (stride=4, 8) tập trung vào small objects. 

Giải pháp:
1. Thêm scale lớn hơn: `stride=[4, 8, 16]`
2. Hoặc dùng `topk` lớn hơn để mỗi object có nhiều anchors hơn
3. Augmentation: Scale jittering để objects lớn vẫn fit vào receptive field

### Q6: Cách debug xem object được gán vào scale nào?
**A:** Thêm logging vào loss function:
```python
# File: ultralytics/utils/loss.py
def get_assigned_targets_and_loss(self, preds, batch):
    ...
    fg_mask = ...  # Shape: (bs, num_anchors)
    
    # Debug: Xem phân bố positive anchors
    if self.debug:
        for i, stride in enumerate(self.stride):
            start = sum(feat.shape[2] * feat.shape[3] for feat in preds["feats"][:i])
            end = start + preds["feats"][i].shape[2] * preds["feats"][i].shape[3]
            count = fg_mask[:, start:end].sum()
            print(f"Stride {stride}: {count} positive anchors")
```

---

## 9. Best Practices

### Chọn số lượng scales:
- **2 scales** ([4,8] hoặc [8,16]): Small object detection, fast inference
- **3 scales** ([8,16,32]): Balanced - recommended
- **4-5 scales**: Very dense detection, slow but accurate

### Chọn stride values:
- **Stride=4**: Rất nhỏ, cho objects < 32px
- **Stride=8**: Nhỏ, 32-64px
- **Stride=16**: Trung bình, 64-128px  
- **Stride=32**: Lớn, 128-256px
- **Stride=64**: Rất lớn, >256px

### Điều chỉnh hyperparameters:
```python
# Với config có nhiều anchors (stride nhỏ)
tal_topk = 15-20  # Tăng từ 10-13
batch_size = 8-16  # Giảm để fit memory
imgsz = 512-640    # Có thể giảm nếu cần

# Với config có ít anchors (stride lớn)  
tal_topk = 8-10    # Giảm xuống
batch_size = 32-64 # Tăng lên
```

---

**Tham khảo thêm:**
- Paper: [TOOD: Task-aligned One-stage Object Detection](https://arxiv.org/abs/2108.07755)
- Code: [`ultralytics/utils/tal.py`](ultralytics/utils/tal.py)
- Code: [`ultralytics/utils/loss.py`](ultralytics/utils/loss.py)
- Code: [`ultralytics/nn/tasks.py`](ultralytics/nn/tasks.py) - Stride computation
- Your config: [`ultralytics/cfg/models/15/yoloav1.yaml`](ultralytics/cfg/models/15/yoloav1.yamlhors

# Giả lập predictions - 2 scales với stride nhỏ hơn
feats = [
    torch.randn(1, 128, 160, 160),  # P2 - stride 4 (higher resolution!)
    torch.randn(1, 256, 80, 80),    # P3 - stride 8
]
strides = torch.tensor([4, 8])

# Tạo anchors
anchor_points, stride_tensor = make_anchors(feats, strides, 0.5)
print(f"Total anchors: {anchor_points.shape[0]}")  # 32,000 (25,600 + 6,400)
print(f"Strides: {strides.tolist()}")
print(f"Anchor breakdown:")
print(f"  - Scale stride=4: 160×160 = 25,600 anchors")
print(f"  - Scale stride=8: 80×80   = 6,400 anchors")

# Giả lập GT: Small object - cup
gt_labels = torch.tensor([[[39]]])  # class 39 (cup)
gt_bboxes = torch.tensor([[[450, 450, 480, 480]]])  # small 30×30 object
mask_gt = torch.ones(1, 1, 1)

# Predictions
pred_scores = torch.randn(1, 32000, 80).sigmoid()
pred_bboxes = torch.randn(1, 32000, 4)

# Assignment
assigner = TaskAlignedAssigner(topk=13, num_classes=80, stride=strides.tolist())
_, target_bboxes, target_scores, fg_mask, target_gt_idx = assigner(
    pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt
)

# Kết quả
print(f"\nAssignment results:")
print(f"Positive anchors: {fg_mask.sum().item()}")  # ~13
print(f"Negative anchors: {(~fg_mask).sum().item()}")  # ~31,987

# Xem phân bố anchors theo scale
positive_indices = torch.where(fg_mask[0])[0]
scale1_count = (positive_indices < 25600).sum().item()
scale2_count = (positive_indices >= 25600).sum().item()
print(f"\nPositive anchor distribution:")
print(f"  - Stride=4 (P2): {scale1_count} anchors")
print(f"  - Stride=8 (P3): {scale2_count} anchors")
print(f"→ Small object được detect chủ yếu ở stride=4!")
```

### 7.3. So sánh Performance

```python
# Benchmark với different configs
import time

configs = {
    "Standard (3 scales)": ([8, 16, 32], [(80,80), (40,40), (20,20)]),
    "Your Config (2 scales)": ([4, 8], [(160,160), (80,80)]),
    "Dense (4 scales)": ([4, 8, 16, 32], [(160,160), (80,80), (40,40), (20,20)]),
}

for name, (strides, shapes) in configs.items():
    feats = [torch.randn(1, 64, h, w) for h, w in shapes]
    
    start = time.time()
    anchor_points, _ = make_anchors(feats, torch.tensor(strides), 0.5)
    elapsed = (time.time() - start) * 1000
    
    print(f"\n{name}:")
    print(f"  Strides: {strides}")
    print(f"  Total anchors: {anchor_points.shape[0]:,}")
    print(f"  Make anchors time: {elapsed:.2f}ms
# Xem anchors nào được gán
positive_anchor_indices = torch.where(fg_mask[0])[0]
print(f"Assigned anchor indices: {positive_anchor_indices}")
```

---

**Tham khảo thêm:**
- Paper: [TOOD: Task-aligned One-stage Object Detection](https://arxiv.org/abs/2108.07755)
- Code: [`ultralytics/utils/tal.py`](ultralytics/utils/tal.py)
- Code: [`ultralytics/utils/loss.py`](ultralytics/utils/loss.py)
