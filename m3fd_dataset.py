"""
シンプルM3FDデータセット - 検出専用版 (Mosaic長方形対応・完全修正版)
RGBとIRを同期させてMosaic/MixUp拡張を行う強化版
修正点:
1. Mosaic処理を長方形 (640x512) に対応させ、ラベルズレを解消
2. クラスIDエラー回避も維持
"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import os
import glob
from PIL import Image
import random 
import logging
import cv2
from typing import Dict, List, Tuple, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleM3FDDataset(data.Dataset):
    def __init__(self, config: Dict[str, Any], mode: str = 'train', augmentation: bool = True, max_samples: Optional[int] = None): 
        self.config = config
        self.mode = mode
        self.augmentation = augmentation and (mode == 'train')
        self.max_samples = max_samples
        
        dataset_config = config['dataset']
        self.img_width = dataset_config['img_width']
        self.img_height = dataset_config['img_height']
        # self.image_size は (H, W)
        self.image_size = (self.img_height, self.img_width)
        self.num_classes = dataset_config['num_classes']
        
        if mode == 'train':
            data_config = dataset_config['train']
        else:
            data_config = dataset_config['validation']

        self.rgb_dir = data_config['rgb_dir']
        self.ir_dir = data_config['ir_dir']
        self.label_dir = data_config['label_dir']
        
        if self.max_samples is None:
            self.max_samples = data_config.get('max_samples', None)

        self.transform_prob = dataset_config.get('transform_prob', 0.5)
        self.mosaic_prob = dataset_config.get('mosaic_prob', 0.5)
        self.mixup_prob = dataset_config.get('mixup_prob', 0.1)

        # クラスIDオフセット: アノテーションが1始まりなら1、0始まりなら0
        self.class_id_offset = dataset_config.get('class_id_offset', 0)
        
        self.data_quality_stats = {
            'total_samples': 0, 'valid_samples': 0,
            'samples_with_objects': 0, 'total_objects': 0
        }
        
        self.setup_transforms()
        self.samples = self._load_and_validate_dataset()

        # クラス別オブジェクト数をカウント
        self.class_counts = self._count_class_distribution()

        logger.info(f"Simple M3FD Dataset initialized:")
        logger.info(f"   Mode: {mode}, Samples: {len(self.samples)}")
        logger.info(f"   Augmentation: {self.augmentation} (Mosaic: {self.mosaic_prob}, MixUp: {self.mixup_prob})")
        logger.info(f"   Target Size: {self.img_width}x{self.img_height}")
        logger.info(f"   Class ID Offset: {self.class_id_offset}")
        logger.info(f"   Class Distribution: {self.class_counts}")

    def _count_class_distribution(self) -> Dict[str, int]:
        """各クラスのオブジェクト数をカウント"""
        class_names = self.config['dataset'].get('class_names',
                                                  [f'class_{i}' for i in range(self.num_classes)])
        counts = {name: 0 for name in class_names}

        for sample in self.samples:
            label_path = sample['label_path']
            if not os.path.exists(label_path):
                continue
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            cls = int(float(parts[0])) - self.class_id_offset
                            if 0 <= cls < len(class_names):
                                counts[class_names[cls]] += 1
            except:
                continue
        return counts

    def setup_transforms(self):
        self.to_tensor = transforms.ToTensor()
        self.rgb_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.ir_normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    def _load_and_validate_dataset(self) -> List[Dict[str, Any]]:
        all_rgb_files = []
        for dir_name, dir_path in [('RGB', self.rgb_dir), ('IR', self.ir_dir), ('Labels', self.label_dir)]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"{dir_name} directory not found: {dir_path}")
        
        rgb_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for pattern in rgb_patterns:
            all_rgb_files.extend(glob.glob(os.path.join(self.rgb_dir, pattern))) 
        
        if not all_rgb_files:
            raise FileNotFoundError(f"No RGB images found in {self.rgb_dir}")
        
        files_to_process = all_rgb_files
        if self.max_samples is not None and 0 < self.max_samples < len(all_rgb_files):
            random.seed(self.config.get('seed', 42))
            files_to_process = random.sample(all_rgb_files, self.max_samples)
        
        files_to_process = sorted(files_to_process)
        samples = []
        
        for rgb_path in files_to_process: 
            try:
                base_name = os.path.splitext(os.path.basename(rgb_path))[0]
                ir_path = self._find_corresponding_file(self.ir_dir, base_name)
                label_path = self._find_corresponding_file(self.label_dir, base_name, ext='.txt')
                
                if ir_path and label_path:
                    samples.append({
                        'image_id': base_name,
                        'rgb_path': rgb_path,
                        'ir_path': ir_path,
                        'label_path': label_path
                    })
            except Exception:
                continue
                
        return samples

    def _find_corresponding_file(self, directory: str, base_name: str, ext: Optional[str] = None) -> Optional[str]:
        if ext:
            candidates = [os.path.join(directory, base_name + ext)]
        else:
            candidates = [os.path.join(directory, base_name + ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']]
        for c in candidates:
            if os.path.exists(c): return c
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def load_image_pair(self, index):
        sample = self.samples[index]
        rgb = cv2.imread(sample['rgb_path'])
        if rgb is None: raise FileNotFoundError(f"Image not found: {sample['rgb_path']}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        ir = cv2.imread(sample['ir_path'], cv2.IMREAD_GRAYSCALE)
        if ir is None: raise FileNotFoundError(f"Image not found: {sample['ir_path']}")
        
        h, w = rgb.shape[:2]
        if ir.shape[:2] != (h, w):
            ir = cv2.resize(ir, (w, h))
            
        if ir.ndim == 2:
            ir = ir[:, :, np.newaxis]
        
        return rgb, ir, (h, w), sample

    def load_mosaic(self, index):
        """Mosaic Augmentation (長方形対応版)"""
        indices = [index] + random.choices(range(len(self.samples)), k=3)
        random.shuffle(indices)
        
        # ✅ 修正: 縦と横を別々に扱う
        w_target = self.img_width
        h_target = self.img_height
        
        # キャンバスの作成 (入力サイズの2倍)
        # H, W の順であることに注意
        result_rgb = np.full((h_target * 2, w_target * 2, 3), 114, dtype=np.uint8)
        result_ir = np.full((h_target * 2, w_target * 2, 1), 114, dtype=np.uint8)
        result_labels = []
        
        # 中心の分割点をランダムに決定 (長方形対応)
        # xは幅の0.5~1.5倍、yは高さの0.5~1.5倍の位置
        xc = int(random.uniform(w_target * 0.5, w_target * 1.5))
        yc = int(random.uniform(h_target * 0.5, h_target * 1.5))
        
        for i, idx in enumerate(indices):
            rgb, ir, (h, w), sample = self.load_image_pair(idx)
            labels = self._load_labels_pixel(sample['label_path'], w, h)
            
            # 画像の配置位置計算
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, w_target * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, h_target * 2)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, w_target * 2), min(yc + h, h_target * 2)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            # 画像貼り付け
            result_rgb[y1a:y2a, x1a:x2a] = rgb[y1b:y2b, x1b:x2b]
            result_ir[y1a:y2a, x1a:x2a] = ir[y1b:y2b, x1b:x2b]
            
            # ラベル座標変換
            padw = x1a - x1b
            padh = y1a - y1b
            
            if labels is not None and len(labels) > 0:
                labels[:, 1] += padw
                labels[:, 2] += padh
                labels[:, 3] += padw
                labels[:, 4] += padh
                result_labels.append(labels)
                
        # ラベル結合とクリップ
        if len(result_labels) > 0:
            result_labels = np.concatenate(result_labels, 0)
            # キャンバスサイズ内にクリップ
            np.clip(result_labels[:, 1::2], 0, 2 * w_target, out=result_labels[:, 1::2]) # x
            np.clip(result_labels[:, 2::2], 0, 2 * h_target, out=result_labels[:, 2::2]) # y
        else:
            result_labels = np.zeros((0, 5))

        # 指定サイズにリサイズ (2W, 2H) -> (W, H) なので正確に0.5倍
        result_rgb = cv2.resize(result_rgb, (w_target, h_target))
        result_ir = cv2.resize(result_ir, (w_target, h_target))
        
        if result_ir.ndim == 2:
            result_ir = result_ir[:, :, np.newaxis]
        
        # ラベルも0.5倍
        if len(result_labels) > 0:
            result_labels[:, 1:] *= 0.5
            
        return result_rgb, result_ir, result_labels

    def mixup(self, rgb1, ir1, labels1, rgb2, ir2, labels2):
        r = np.random.beta(32.0, 32.0)
        rgb = (rgb1 * r + rgb2 * (1 - r)).astype(np.uint8)
        ir = (ir1 * r + ir2 * (1 - r)).astype(np.uint8)
        
        if len(labels1) > 0 and len(labels2) > 0:
            labels = np.concatenate((labels1, labels2), 0)
        elif len(labels1) > 0:
            labels = labels1
        else:
            labels = labels2
            
        return rgb, ir, labels

    def _load_labels_pixel(self, label_path, w, h):
        if not os.path.exists(label_path): return np.zeros((0, 5))
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            labels = []
            for line in lines:
                parts = list(map(float, line.strip().split()))
                if len(parts) != 5: continue
                cls, x1, y1, x2, y2 = parts

                # クラスID補正（0始まりに変換）
                # class_id_offset=1: アノテーションが1始まり（1,2,3...）→ 0始まりに変換
                # class_id_offset=0: アノテーションが0始まり（0,1,2...）→ そのまま
                cls = int(cls) - self.class_id_offset

                if not (0 <= cls < self.num_classes):
                    continue
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1: continue
                
                labels.append([cls, x1, y1, x2, y2])
            return np.array(labels)
        except:
            return np.zeros((0, 5))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        safe_idx = idx % len(self.samples)
        
        if self.augmentation and random.random() < self.mosaic_prob:
            rgb_img, ir_img, labels = self.load_mosaic(safe_idx)
            
            if random.random() < self.mixup_prob:
                idx2 = random.randint(0, len(self.samples) - 1)
                rgb2, ir2, labels2 = self.load_mosaic(idx2)
                rgb_img, ir_img, labels = self.mixup(rgb_img, ir_img, labels, rgb2, ir2, labels2)
                
        else:
            rgb_img, ir_img, (h, w), sample = self.load_image_pair(safe_idx)
            labels = self._load_labels_pixel(sample['label_path'], w, h)
            
            rgb_img = cv2.resize(rgb_img, (self.img_width, self.img_height))
            ir_img = cv2.resize(ir_img, (self.img_width, self.img_height))
            
            if ir_img.ndim == 2:
                ir_img = ir_img[:, :, np.newaxis]
            
            if len(labels) > 0:
                labels[:, [1, 3]] *= (self.img_width / w)
                labels[:, [2, 4]] *= (self.img_height / h)

            if self.augmentation and random.random() < 0.5:
                rgb_img = cv2.flip(rgb_img, 1)
                ir_img = cv2.flip(ir_img, 1)
                if ir_img.ndim == 2:
                    ir_img = ir_img[:, :, np.newaxis]
                if len(labels) > 0:
                    labels[:, 1] = self.img_width - labels[:, 1]
                    labels[:, 3] = self.img_width - labels[:, 3]
                    labels[:, [1, 3]] = labels[:, [3, 1]]

        rgb_tensor = self.to_tensor(Image.fromarray(rgb_img))
        ir_tensor = self.to_tensor(Image.fromarray(ir_img[:,:,0]))

        rgb_tensor = self.rgb_normalize(rgb_tensor)
        ir_tensor = self.ir_normalize(ir_tensor)

        # IRコントラスト・明るさ調整（訓練時のみ）
        if self.augmentation:
            # コントラスト調整（IR - 強化版）
            if random.random() < 0.5:  # 0.3 -> 0.5
                contrast_factor = random.uniform(0.7, 1.4)  # 0.8-1.2 -> 0.7-1.4
                ir_tensor = TF.adjust_contrast(ir_tensor, contrast_factor)

            # IRの明るさ調整（追加）
            if random.random() < 0.4:
                brightness_factor = random.uniform(0.8, 1.2)
                ir_tensor = TF.adjust_brightness(ir_tensor, brightness_factor)
        
        final_targets = []
        if len(labels) > 0:
            for box in labels:
                cls, x1, y1, x2, y2 = box
                w_box = x2 - x1
                h_box = y2 - y1
                
                if w_box < 2 or h_box < 2: continue
                
                x_center = (x1 + x2) / 2 / self.img_width
                y_center = (y1 + y2) / 2 / self.img_height
                w_norm = w_box / self.img_width
                h_norm = h_box / self.img_height
                
                if not (0 <= int(cls) < self.num_classes): continue

                x_center = np.clip(x_center, 0, 1)
                y_center = np.clip(y_center, 0, 1)
                w_norm = np.clip(w_norm, 0, 1)
                h_norm = np.clip(h_norm, 0, 1)
                
                final_targets.append([cls, x_center, y_center, w_norm, h_norm])
        
        if len(final_targets) > 0:
            target_tensor = torch.tensor(final_targets, dtype=torch.float32)
        else:
            target_tensor = torch.full((1, 5), -1.0, dtype=torch.float32)

        return {
            'ir_image': ir_tensor,
            'rgb_image': rgb_tensor,
            'detection_gt': target_tensor,
            'image_id': self.samples[safe_idx]['image_id'],
            'ir_path': self.samples[safe_idx]['ir_path'],
            'rgb_path': self.samples[safe_idx]['rgb_path']
        }

class SimpleCollateFunction:
    def __init__(self, max_objects: int = 120):
        self.max_objects = max_objects
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        valid_batch = [item for item in batch if item is not None]
        if not valid_batch: return None
        
        ir_images = torch.stack([item['ir_image'] for item in valid_batch])
        rgb_images = torch.stack([item['rgb_image'] for item in valid_batch])
        
        detection_targets = []
        for item in valid_batch:
            targets = item['detection_gt']
            if targets.dim() > 1 and targets.size(1) == 5:
                valid_mask = targets[:, 0] >= 0
                targets = targets[valid_mask]
            
            if len(targets) > 0:
                if len(targets) > self.max_objects:
                    targets = targets[:self.max_objects]
                elif len(targets) < self.max_objects:
                    padding = torch.full((self.max_objects - len(targets), 5), -1.0)
                    targets = torch.cat([targets, padding], dim=0)
            else:
                targets = torch.full((self.max_objects, 5), -1.0)
            
            detection_targets.append(targets)
        
        return {
            'ir_image': ir_images,
            'rgb_image': rgb_images,
            'detection_gt': torch.stack(detection_targets),
            'batch_size': len(valid_batch)
        }

def create_simple_m3fd_data_loaders(config: Dict[str, Any]) -> Dict[str, data.DataLoader]:
    dataset_config = config['dataset']
    train_max_samples = dataset_config['train'].get('max_samples', None)
    val_max_samples = dataset_config['validation'].get('max_samples', None)

    train_dataset = SimpleM3FDDataset(config, mode='train', augmentation=True, max_samples=train_max_samples)
    val_dataset = SimpleM3FDDataset(config, mode='val', augmentation=False, max_samples=val_max_samples)
    
    collate_fn = SimpleCollateFunction(max_objects=120)
    
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=dataset_config['batch_size'],
        shuffle=True,
        num_workers=dataset_config.get('num_workers', 0),
        collate_fn=collate_fn,
        pin_memory=dataset_config.get('pin_memory', False),
        drop_last=True
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=dataset_config['batch_size'],
        shuffle=False,
        num_workers=dataset_config.get('num_workers', 0),
        collate_fn=collate_fn,
        pin_memory=dataset_config.get('pin_memory', False),
        drop_last=False
    )
    
    return {'train': train_loader, 'val': val_loader}

if __name__ == "__main__":
    pass