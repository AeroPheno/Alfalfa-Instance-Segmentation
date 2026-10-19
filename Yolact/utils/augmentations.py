import os
import math
import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import cv2
import types
from numpy import random
from math import sqrt


# 内存保护：
_MAX_MASK_BYTES = int(os.environ.get("YOLACT_MAX_MASK_BYTES", str(1024 * 1024 * 1024)))


def _estimate_mask_bytes(num_masks: int, h: int, w: int, dtype) -> int:
    """估算 masks 占用字节数（不含额外中间拷贝）。"""
    try:
        return int(num_masks) * int(h) * int(w) * np.dtype(dtype).itemsize
    except Exception:
        return int(num_masks) * int(h) * int(w)

try:
    from data import cfg, MEANS, STD
except ImportError:
    MEANS = (103.94, 116.78, 123.68)
    STD = (1, 1, 1)


    class MockCfg:
        max_size = 1024
        preserve_aspect_ratio = True
        discard_box_width = 1
        discard_box_height = 1
        use_gt_bboxes = False
        mask_size = 16
        augment_photometric_distort = True
        augment_expand = True
        augment_random_sample_crop = True
        augment_random_mirror = True
        augment_random_flip = False
        backbone = type('obj', (object,), {'transform': type('obj', (object,), {
            'normalize': True,
            'subtract_means': False,
            'to_float': False,
            'channel_order': 'RGB'
        })})


    cfg = MockCfg()


# --------------------------
# 核心修复1：通用安全resize函数（解决!dsize.empty()断言+空图像/掩码问题）
# --------------------------
def safe_cv2_resize(src, dsize, interpolation=cv2.INTER_LINEAR):
    """安全的cv2.resize封装，避免尺寸异常导致的崩溃"""
    # 1. 校验dsize有效性
    if not dsize or len(dsize) != 2:
        print(f"⚠️ 无效的dsize参数：{dsize}，使用原尺寸兜底")
        return src

    width, height = dsize
    # 2. 确保尺寸为正整数
    width = max(1, int(width))
    height = max(1, int(height))

    # 3. 校验原图像尺寸（避免空输入）
    if src.size == 0 or len(src.shape) < 2:
        print(f"⚠️ 输入图像/掩码为空，尺寸：{src.shape if len(src.shape) >= 2 else '未知'}")
        return np.zeros((height, width), dtype=src.dtype) if len(src.shape) == 2 else np.zeros(
            (height, width, src.shape[2]), dtype=src.dtype)

    # 4. 执行安全resize
    try:
        return cv2.resize(src, (width, height), interpolation=interpolation)
    except Exception as e:
        print(f"⚠️ resize失败：{e}，使用原尺寸兜底")
        return src


# --------------------------
# 基础增强类（无核心修改，保留）
# --------------------------
class BaseTransform:
    def __call__(self, img, mask=None):
        img = np.array(img)
        return img, mask


class SSDAugmentation:
    def __call__(self, img, target=None):
        img = np.array(img)
        return img, target


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes."""
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union


class Compose(object):
    """Composes several augmentations together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, masks=None, boxes=None, labels=None):
        for t in self.transforms:
            img, masks, boxes, labels = t(img, masks, boxes, labels)
        return img, masks, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, masks=None, boxes=None, labels=None):
        return self.lambd(img, masks, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, masks=None, boxes=None, labels=None):
        return image.astype(np.float32), masks, boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, masks=None, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        return image, masks, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, masks=None, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height
        return image, masks, boxes, labels


# --------------------------
# 修复2：Pad类（解决超大图像/掩码越界+维度不匹配）
# --------------------------
class Pad(object):
    def __init__(self, width, height, mean=MEANS, pad_gt=True):
        self.mean = mean
        self.width = width
        self.height = height
        self.pad_gt = pad_gt

    def __call__(self, image, masks, boxes=None, labels=None):
        im_h, im_w, depth = image.shape

        # 处理超大尺寸图像（先缩放到目标尺寸）
        if im_w > self.width or im_h > self.height:
            scale = min(self.width / im_w, self.height / im_h)
            new_w = int(im_w * scale)
            new_h = int(im_h * scale)
            image = safe_cv2_resize(image, (new_w, new_h))
            im_h, im_w = new_h, new_w

        # 初始化填充后的图像
        expand_image = np.zeros((self.height, self.width, depth), dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[:im_h, :im_w] = image

        if self.pad_gt and masks is not None:
            # 掩码维度标准化（确保是3维 [num_masks, h, w]）
            if len(masks.shape) == 2:
                masks = np.expand_dims(masks, 0)
            num_masks = masks.shape[0]

            # 初始化填充后的掩码
            expand_masks = np.zeros((num_masks, self.height, self.width), dtype=masks.dtype)

            # 缩放掩码到图像尺寸
            if masks.shape[1:] != (im_h, im_w):
                scaled_masks = []
                for i in range(num_masks):
                    scaled_mask = safe_cv2_resize(masks[i], (im_w, im_h))
                    scaled_masks.append(scaled_mask)
                masks = np.stack(scaled_masks, axis=0)

            # 安全赋值（避免越界）
            mask_h = min(masks.shape[1], self.height)
            mask_w = min(masks.shape[2], self.width)
            expand_masks[:, :mask_h, :mask_w] = masks[:, :mask_h, :mask_w]
            masks = expand_masks

        return expand_image, masks, boxes, labels


# --------------------------
# 核心修复3：Resize类（解决布尔索引维度不匹配+索引越界）
# --------------------------
class Resize(object):
    @staticmethod
    def calc_size_preserve_ar(img_w, img_h, max_size):
        if img_h <= 0 or img_w <= 0:
            return max_size, max_size
        ratio = sqrt(max(img_w / img_h, 1e-6))
        w = max_size * ratio
        h = max_size / ratio
        w = max(1, min(int(w), max_size * 2))
        h = max(1, min(int(h), max_size * 2))
        return w, h

    def __init__(self, resize_gt=True):
        self.resize_gt = resize_gt
        self.max_size = cfg.max_size
        self.preserve_aspect_ratio = cfg.preserve_aspect_ratio

    def __call__(self, image, masks, boxes, labels=None):
        img_h, img_w, _ = image.shape

        # 校验原始图像尺寸有效性
        if img_h <= 0 or img_w <= 0:
            print(f"⚠️ 无效的原始图像尺寸({img_w}x{img_h})，使用默认尺寸")
            img_w, img_h = self.max_size, self.max_size
            image = np.zeros((img_h, img_w, 3), dtype=image.dtype) + MEANS

        # 计算目标尺寸
        if self.preserve_aspect_ratio:
            width, height = Resize.calc_size_preserve_ar(img_w, img_h, self.max_size)
        else:
            width, height = self.max_size, self.max_size

        # 安全缩放图像
        image = safe_cv2_resize(image, (width, height))

        if self.resize_gt and masks is not None and boxes is not None:
            # 掩码维度标准化
            if len(masks.shape) == 2:
                masks = np.expand_dims(masks, 0)

            # 缩放掩码（逐实例resize，避免 OpenCV 通道数上限导致的异常与错误拷贝）
            if (img_w != width) or (img_h != height):
                num_masks = masks.shape[0]
                resized_masks = np.empty((num_masks, height, width), dtype=masks.dtype)
                for i in range(num_masks):
                    resized_masks[i] = safe_cv2_resize(masks[i], (width, height), interpolation=cv2.INTER_NEAREST)
                masks = resized_masks

            # 缩放框（校验比例有效性）
            if img_w > 0 and img_h > 0:
                boxes[:, [0, 2]] *= (width / img_w)
                boxes[:, [1, 3]] *= (height / img_h)
            else:
                boxes[:, [0, 2]] *= 1.0
                boxes[:, [1, 3]] *= 1.0

            # --------------------------
            # 核心修复：维度匹配的索引逻辑
            # --------------------------
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            keep = (w > cfg.discard_box_width) & (h > cfg.discard_box_height)

            # 维度不匹配时：只做安全截断，禁止 repeat/pad（repeat会直接制造超大内存）
            if isinstance(masks, np.ndarray) and masks.shape[0] != boxes.shape[0]:
                min_len = min(masks.shape[0], boxes.shape[0])
                print(f"masks数量({masks.shape[0]})与boxes数量({boxes.shape[0]})不匹配，截断到{min_len}")
                masks = masks[:min_len]
                boxes = boxes[:min_len]
                keep = keep[:min_len]
                if labels is not None and isinstance(labels, dict) and 'labels' in labels:
                    labels_np = np.array(labels['labels'])[:min_len]
                    labels['labels'] = labels_np.tolist()
                    labels['num_crowds'] = (labels_np < 0).sum()

            # 安全索引（兼容数组/列表，避免越界）
            keep_idx = np.where(keep)[0].tolist() if isinstance(keep, np.ndarray) else keep
            masks = masks[keep] if isinstance(masks, np.ndarray) else [masks[i] for i in keep_idx if i < len(masks)]
            boxes = boxes[keep] if isinstance(boxes, np.ndarray) else [boxes[i] for i in keep_idx if i < len(boxes)]

            # 处理labels字典（维度匹配）
            if labels is not None and 'labels' in labels:
                labels_np = np.array(labels['labels'])
                # 确保labels维度与keep匹配
                if len(labels_np) != len(keep):
                    labels_np = labels_np[:len(keep)] if len(labels_np) > len(keep) else np.pad(labels_np, (0,
                                                                                                            len(keep) - len(
                                                                                                                labels_np)),
                                                                                                mode='constant')
                labels['labels'] = labels_np[keep].tolist()
                labels['num_crowds'] = (np.array(labels['labels']) < 0).sum()

        return image, masks, boxes, labels


# --------------------------
# 光度畸变相关类（无核心修改，保留）
# --------------------------
class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower
        assert self.lower >= 0

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
        return image, masks, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, masks, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))

    def __call__(self, image, masks=None, boxes=None, labels=None):
        return image, masks, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, masks, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower
        assert self.lower >= 0

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, masks, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0 and delta <= 255.0
        self.delta = delta

    def __call__(self, image, masks=None, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, masks, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, masks=None, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), masks, boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, masks=None, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), masks, boxes, labels


# --------------------------
# 修复4：RandomSampleCrop类（解决索引越界+维度不匹配）
# --------------------------
class RandomSampleCrop(object):
    def __init__(self,
                 min_crop_ratio=0.3,
                 max_crop_ratio=1.0,
                 min_aspect_ratio=0.5,
                 max_aspect_ratio=2.0,
                 min_area=0.1,
                 max_area=1.0,
                 random_brightness=True,
                 random_contrast=True,
                 random_expand=True):

        self.min_crop_ratio = min_crop_ratio
        self.max_crop_ratio = max_crop_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_area = min_area
        self.max_area = max_area
        self.random_brightness = random_brightness
        self.random_contrast = random_contrast
        self.random_expand = random_expand

    def __call__(self, img, masks=None, boxes=None, labels=None):
        h, w, _ = img.shape

        for _ in range(50):
            scale = random.uniform(self.min_crop_ratio, self.max_crop_ratio)
            aspect = random.uniform(self.min_aspect_ratio, self.max_aspect_ratio)

            crop_h = int(h * scale / math.sqrt(aspect))
            crop_w = int(w * scale * math.sqrt(aspect))

            # 校验裁剪尺寸有效性（避免0/负数/越界）
            if crop_h <= 0 or crop_w <= 0 or crop_h >= h or crop_w >= w:
                continue

            x = random.randint(0, w - crop_w)
            y = random.randint(0, h - crop_h)
            rect = np.array([[x, y, x + crop_w, y + crop_h]])

            # 裁剪图像
            cropped_img = img[y:y + crop_h, x:x + crop_w, :]

            # 裁剪框
            if boxes is not None:
                if masks is not None:
                    if isinstance(masks, np.ndarray):
                        if masks.ndim == 2:
                            masks = np.expand_dims(masks, 0)
                        if masks.shape[0] != boxes.shape[0]:
                            min_len = min(int(masks.shape[0]), int(boxes.shape[0]))
                            print(f"masks数量({masks.shape[0]})与boxes数量({boxes.shape[0]})不匹配，截断到{min_len}")
                            masks = masks[:min_len]
                            boxes = boxes[:min_len]
                            if labels is not None:
                                if isinstance(labels, dict) and "labels" in labels:
                                    labels["labels"] = np.array(labels["labels"])[:min_len].tolist()
                                    labels["num_crowds"] = (np.array(labels["labels"]) < 0).sum()
                                else:
                                    labels = (np.array(labels)[:min_len]).tolist()
                    else:
                        # list masks
                        if len(masks) != len(boxes):
                            min_len = min(len(masks), len(boxes))
                            print(f"masks数量({len(masks)})与boxes数量({len(boxes)})不匹配，截断到{min_len}")
                            masks = masks[:min_len]
                            boxes = boxes[:min_len]
                            if labels is not None:
                                if isinstance(labels, dict) and "labels" in labels:
                                    labels["labels"] = np.array(labels["labels"])[:min_len].tolist()
                                    labels["num_crowds"] = (np.array(labels["labels"]) < 0).sum()
                                else:
                                    labels = (np.array(labels)[:min_len]).tolist()

                new_boxes = boxes.copy()
                new_boxes[:, 0] = np.clip(new_boxes[:, 0] - x, 0, crop_w)
                new_boxes[:, 1] = np.clip(new_boxes[:, 1] - y, 0, crop_h)
                new_boxes[:, 2] = np.clip(new_boxes[:, 2] - x, 0, crop_w)
                new_boxes[:, 3] = np.clip(new_boxes[:, 3] - y, 0, crop_h)

                # 移除全失效的框
                keep = (new_boxes[:, 2] > new_boxes[:, 0]) & (new_boxes[:, 3] > new_boxes[:, 1])
                if not keep.any():
                    continue

                # 安全索引（过滤越界索引）
                keep_idx = np.where(keep)[0].tolist()
                keep_idx = [i for i in keep_idx if i < len(new_boxes)]

                new_boxes = new_boxes[keep] if isinstance(new_boxes, np.ndarray) else [new_boxes[i] for i in keep_idx]

                # 处理labels（维度匹配）
                new_labels = None
                if labels is not None:
                    if isinstance(labels, dict) and 'labels' in labels:
                        labels_np = np.array(labels['labels'])
                        # 维度匹配校验
                        labels_np = labels_np[:len(keep)] if len(labels_np) > len(keep) else labels_np
                        new_label_vals = labels_np[keep].tolist()
                        new_labels = {
                            'labels': new_label_vals,
                            'num_crowds': (np.array(new_label_vals) < 0).sum()
                        }
                    else:
                        labels_np = np.array(labels) if not isinstance(labels, np.ndarray) else labels
                        new_labels = labels_np[keep].tolist()

                # 处理masks（维度匹配+越界过滤）
                new_masks = None
                if masks is not None:
                    if isinstance(masks, np.ndarray):
                        if masks.ndim == 2:
                            masks = np.expand_dims(masks, 0)

                        cropped_masks = masks[:, y:y + crop_h, x:x + crop_w]
                        keep_count = int(np.sum(keep))
                        est = _estimate_mask_bytes(keep_count, crop_h, crop_w, cropped_masks.dtype)
                        if est > _MAX_MASK_BYTES:
                            continue
                        new_masks = cropped_masks[keep]
                    else:
                        # list masks：同样先裁剪再筛选
                        new_masks = [masks[i][y:y + crop_h, x:x + crop_w] for i in keep_idx if i < len(masks)]

                return cropped_img, new_masks, new_boxes, new_labels

        return img, masks, boxes, labels


# --------------------------
# 其他增强类（无核心修改，保留）
# --------------------------
class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, masks, boxes, labels):
        if random.randint(2):
            return image, masks, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        new_height = max(1, int(height * ratio))
        new_width = max(1, int(width * ratio))

        left = random.uniform(0, new_width - width)
        top = random.uniform(0, new_height - height)

        # 内存保护
        if masks is not None:
            num_masks = masks.shape[0] if hasattr(masks, "shape") and len(masks.shape) == 3 else 1
            dtype = masks.dtype if hasattr(masks, "dtype") else np.uint8
            est = _estimate_mask_bytes(num_masks, new_height, new_width, dtype)
            if est > _MAX_MASK_BYTES:
                return image, masks, boxes, labels

        expand_image = np.zeros((new_height, new_width, depth), dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height), int(left):int(left + width)] = image
        image = expand_image

        if masks is not None:
            # 掩码维度标准化
            if len(masks.shape) == 2:
                masks = np.expand_dims(masks, 0)
            expand_masks = np.zeros((masks.shape[0], new_height, new_width), dtype=masks.dtype)
            expand_masks[:, int(top):int(top + height), int(left):int(left + width)] = masks
            masks = expand_masks

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, masks, boxes, labels


class RandomMirror(object):
    def __call__(self, image, masks, boxes, labels):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            if masks is not None:
                if isinstance(masks, np.ndarray):
                    masks = masks[:, :, ::-1]
                else:
                    masks = [m[:, ::-1] for m in masks]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, masks, boxes, labels


class RandomFlip(object):
    def __call__(self, image, masks, boxes, labels):
        height, _, _ = image.shape
        if random.randint(2):
            image = image[::-1, :]
            if masks is not None:
                if isinstance(masks, np.ndarray):
                    masks = masks[:, ::-1, :]
                else:
                    masks = [m[::-1, :] for m in masks]
            boxes = boxes.copy()
            boxes[:, 1::2] = height - boxes[:, 3::-2]
        return image, masks, boxes, labels


class RandomRot90(object):
    def __call__(self, image, masks, boxes, labels):
        old_height, old_width, _ = image.shape
        k = random.randint(4)
        image = np.rot90(image, k)
        if masks is not None:
            if isinstance(masks, np.ndarray):
                masks = np.array([np.rot90(mask, k) for mask in masks])
            else:
                masks = [np.rot90(mask, k) for mask in masks]
        boxes = boxes.copy()
        for _ in range(k):
            boxes = np.array([[box[1], old_width - 1 - box[2], box[3], old_width - 1 - box[0]] for box in boxes])
            old_width, old_height = old_height, old_width
        return image, masks, boxes, labels


class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, masks, boxes, labels):
        im = image.copy()
        im, masks, boxes, labels = self.rand_brightness(im, masks, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, masks, boxes, labels = distort(im, masks, boxes, labels)
        return self.rand_light_noise(im, masks, boxes, labels)


class PrepareMasks(object):
    def __init__(self, mask_size, use_gt_bboxes):
        self.mask_size = mask_size
        self.use_gt_bboxes = use_gt_bboxes

    def __call__(self, image, masks, boxes, labels=None):
        if not self.use_gt_bboxes:
            return image, masks, boxes, labels

        height, width, _ = image.shape
        # 掩码维度标准化
        if len(masks.shape) == 2:
            masks = np.expand_dims(masks, 0)
        new_masks = np.zeros((masks.shape[0], self.mask_size ** 2))

        for i in range(len(masks)):
            x1, y1, x2, y2 = boxes[i, :]
            x1 *= width
            x2 *= width
            y1 *= height
            y2 *= height
            x1, y1, x2, y2 = (max(0, int(x1)), max(0, int(y1)), min(width - 1, int(x2)), min(height - 1, int(y2)))

            # 跳过无效裁剪区域
            if x2 <= x1 or y2 <= y1:
                print(f"⚠️ 无效的掩码裁剪区域：x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                new_masks[i, :] = 0
                continue

            cropped_mask = masks[i, y1:(y2 + 1), x1:(x2 + 1)]
            scaled_mask = safe_cv2_resize(cropped_mask, (self.mask_size, self.mask_size))
            new_masks[i, :] = scaled_mask.reshape(1, -1)

        # 二值化
        new_masks[new_masks > 0.5] = 1
        new_masks[new_masks <= 0.5] = 0

        return image, new_masks, boxes, labels


class BackboneTransform(object):
    def __init__(self, transform, mean, std, in_channel_order):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.transform = transform
        self.channel_map = {c: idx for idx, c in enumerate(in_channel_order)}
        self.channel_permutation = [self.channel_map[c] for c in transform.channel_order]

    def __call__(self, img, masks=None, boxes=None, labels=None):
        img = img.astype(np.float32)

        if self.transform.normalize:
            img = (img - self.mean) / self.std
        elif self.transform.subtract_means:
            img = (img - self.mean)
        elif self.transform.to_float:
            img = img / 255

        img = img[:, :, self.channel_permutation]
        return img.astype(np.float32), masks, boxes, labels


class BaseTransform(object):
    """ Transform to be used when evaluating. """

    def __init__(self, mean=MEANS, std=STD):
        self.augment = Compose([
            ConvertFromInts(),
            Resize(resize_gt=False),
            BackboneTransform(cfg.backbone.transform, mean, std, 'BGR')
        ])

    def __call__(self, img, masks=None, boxes=None, labels=None):
        return self.augment(img, masks, boxes, labels)


class FastBaseTransform(torch.nn.Module):
    """GPU加速的基础变换（生产环境用）"""

    def __init__(self):
        super().__init__()
        self.mean = torch.Tensor(MEANS).float().cuda()[None, :, None, None]
        self.std = torch.Tensor(STD).float().cuda()[None, :, None, None]
        self.transform = cfg.backbone.transform

    def forward(self, img):
        self.mean = self.mean.to(img.device)
        self.std = self.std.to(img.device)

        if cfg.preserve_aspect_ratio:
            _, h, w, _ = img.size()
            img_size = Resize.calc_size_preserve_ar(w, h, cfg.max_size)
            img_size = (img_size[1], img_size[0])
        else:
            img_size = (cfg.max_size, cfg.max_size)

        img = img.permute(0, 3, 1, 2).contiguous()
        img = F.interpolate(img, img_size, mode='bilinear', align_corners=False)

        if self.transform.normalize:
            img = (img - self.mean) / self.std
        elif self.transform.subtract_means:
            img = (img - self.mean)
        elif self.transform.to_float:
            img = img / 255

        if self.transform.channel_order != 'RGB':
            raise NotImplementedError

        img = img[:, (2, 1, 0), :, :].contiguous()
        return img


def do_nothing(img=None, masks=None, boxes=None, labels=None):
    return img, masks, boxes, labels


def enable_if(condition, obj):
    return obj if condition else do_nothing


class SSDAugmentation(object):
    """ Training用的增强组合 """

    def __init__(self, mean=MEANS, std=STD):
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            enable_if(cfg.augment_photometric_distort, PhotometricDistort()),
            enable_if(cfg.augment_expand, Expand(mean)),
            enable_if(cfg.augment_random_sample_crop, RandomSampleCrop()),
            enable_if(cfg.augment_random_mirror, RandomMirror()),
            enable_if(cfg.augment_random_flip, RandomFlip()),
            enable_if(cfg.augment_random_flip, RandomRot90()),
            Resize(),
            enable_if(not cfg.preserve_aspect_ratio, Pad(cfg.max_size, cfg.max_size, mean)),
            ToPercentCoords(),
            PrepareMasks(cfg.mask_size, cfg.use_gt_bboxes),
            BackboneTransform(cfg.backbone.transform, mean, std, 'BGR')
        ])

    def __call__(self, img, masks, boxes, labels):
        return self.augment(img, masks, boxes, labels)