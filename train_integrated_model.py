"""
統合融合検出モデルの訓練スクリプト（TensorBoard可視化強化版）
変更点:
1. TensorBoardに「元画像(RGB/IR)」「融合結果+BBox」「重みマップ画像」を送信する機能を追加
2. 重みの分布(ヒストグラム)を送信する機能を追加
branch: feature/preprocess
"""

import os
import sys
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
import logging
from typing import Dict, Optional, List, Tuple
import argparse
import time
import numpy as np
from copy import deepcopy
from datetime import datetime
import cv2  # OpenCVを使用

# 混合精度訓練用
try:
    from torch.cuda.amp import GradScaler
    if torch.cuda.is_available():
        from torch.cuda.amp import autocast
    else:
        autocast = None
    AMP_AVAILABLE = True
except ImportError:
    autocast = None
    AMP_AVAILABLE = False
    print("Warning: torch.cuda.amp not available, mixed precision training disabled")

# プログレスバー
from tqdm import tqdm

# YOLOv5のパスを追加
YOLO_PATH = Path(__file__).parent / 'yolov5'
if YOLO_PATH.exists():
    sys.path.insert(0, str(YOLO_PATH))
    try:
        from utils.general import (colorstr, check_img_size, 
                                   xyxy2xywh, xywh2xyxy, box_iou, non_max_suppression, 
                                   strip_optimizer)
        from utils.torch_utils import ModelEMA, select_device, de_parallel
        from utils.metrics import ap_per_class, ConfusionMatrix, fitness
        from utils.plots import plot_results, output_to_target
        YOLO_UTILS_AVAILABLE = True
        print(f"YOLOv5: utils loaded successfully from {YOLO_PATH}")
    except ImportError as e:
        YOLO_UTILS_AVAILABLE = False
        def fitness(x): return 0.0
        print(f"Warning: YOLOv5 utils not fully available: {e}")
else:
    YOLO_UTILS_AVAILABLE = False
    def fitness(x): return 0.0
    print(f"Warning: YOLOv5 not found at {YOLO_PATH}")

# モデルとデータセットのインポート
from integrated_fusion_detection_model import create_integrated_model
from m3fd_dataset import create_simple_m3fd_data_loaders, SimpleM3FDDataset

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available")

# matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ロギング設定
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter('%(message)s'))
LOGGER.addHandler(console)

RANK = int(os.getenv('RANK', -1))


class IntegratedModelTrainer:

    def __init__(self, config: Dict, output_dir: str = './runs/train/exp', 
                 device='', add_graph=False):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.add_graph_flag = add_graph
        
        # デバイス設定
        device_config = config.get('device', device if device else '0')
        if device_config == 'cpu':
            self.device = torch.device('cpu')
            LOGGER.info('Using CPU for training (safe but slow)')
            config['dataset']['num_workers'] = 0
        else:
            if torch.cuda.is_available():
                if YOLO_UTILS_AVAILABLE:
                    try:
                        self.device = select_device(device_config, batch_size=config['dataset'].get('batch_size', 2))
                        LOGGER.info(f'Using CUDA device: {self.device}')
                    except AssertionError as e:
                        LOGGER.warning(f'CUDA device selection failed: {e}')
                        LOGGER.info('Falling back to CPU')
                        self.device = torch.device('cpu')
                        config['dataset']['num_workers'] = 0
                else:
                    self.device = torch.device(device_config)
                    LOGGER.info(f'Using CUDA device: {self.device}')
            else:
                LOGGER.warning('CUDA not available, falling back to CPU')
                self.device = torch.device('cpu')
                config['dataset']['num_workers'] = 0
        
        # ディレクトリ設定
        self.save_dir = self.output_dir
        self.wdir = self.save_dir / 'weights'
        self.wdir.mkdir(parents=True, exist_ok=True)
        self.last = self.wdir / 'last.pt'
        self.best = self.wdir / 'best.pt'
        self.results_file = self.save_dir / 'results.txt'
        self.per_class_results_file = self.save_dir / 'per_class_ap.csv'
        self.results_npy_file = self.save_dir / 'results.npy'
        self.per_class_npy_file = self.save_dir / 'per_class_ap.npy'
        
        # ヘッダー
        self.print_train_header()
        
        # ハイパーパラメータ表示
        self.print_hyperparameters()
        
        # モデル作成
        LOGGER.info(f'\nModel: Creating integrated fusion-detection model...')
        self.model = self._create_model()
        
        # クラス設定
        self.num_classes = config['dataset']['num_classes']
        self.class_names = config['dataset'].get('class_names', 
                                                [f'class_{i}' for i in range(self.num_classes)])
        
        # オプティマイザとスケジューラ
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # EMA
        self.ema = None
        if YOLO_UTILS_AVAILABLE:
            try:
                self.ema = ModelEMA(self.model)
                LOGGER.info(f'EMA: Enabled')
            except Exception as e:
                LOGGER.warning(f'EMA failed: {e}')
        
        # データローダー
        LOGGER.info(f'\nDataset: Loading...')
        self.dataloaders = create_simple_m3fd_data_loaders(config)
        self.train_loader = self.dataloaders['train']
        self.val_loader = self.dataloaders['val']
        
        LOGGER.info(f'{"Image sizes":<20}{config["dataset"]["img_width"]} train, {config["dataset"]["img_width"]} val')
        LOGGER.info(f'{"Train batches":<20}{len(self.train_loader)}')
        LOGGER.info(f'{"Val batches":<20}{len(self.val_loader)}')
        
        # 訓練状態
        self.current_epoch = 0
        self.best_fitness = 0.0
        self.start_epoch = 0
        self.results = []
        self.per_class_results = []
        
        # TensorBoard
        self.tb_writer = None
        if TENSORBOARD_AVAILABLE and RANK in {-1, 0}:
            self.tb_writer = SummaryWriter(str(self.save_dir))
            LOGGER.info(f'TensorBoard: Start with \'tensorboard --logdir {self.save_dir.parent}\', '
                       f'view at http://localhost:6006/')
        
        self.use_amp = False
        self.scaler = GradScaler() if self.use_amp else None
        LOGGER.info('AMP: Disabled (MMDetection TOOD incompatible)')
        
        self.accumulation_steps = self.config.get('training', {}).get('accumulation_steps', 1)
        if self.accumulation_steps > 1:
            LOGGER.info(f'Gradient Accumulation: {self.accumulation_steps} steps')
    
    def print_train_header(self):
        import platform
        LOGGER.info('\ntrain: ' + 
                   f'weights={self.wdir}, epochs={self.config["training"]["epochs"]}, '
                   f'batch_size={self.config["dataset"]["batch_size"]}')
        LOGGER.info(f'Python-{platform.python_version()} '
                   f'torch-{torch.__version__} '
                   f'CUDA:{torch.version.cuda if torch.cuda.is_available() else "CPU"}')
        
        if RANK in {-1, 0}:
            LOGGER.info(('\n' + '%11s' * 5) % 
                    ('Epoch', 'GPU_mem', 'box', 'cls', 'total'))
    
    def print_hyperparameters(self):
        train_config = self.config.get('training', {})
        LOGGER.info(f'\nHyperparameters:')
        for k, v in train_config.items():
            LOGGER.info(f'  {k}: {v}')
    
    def _create_model(self) -> nn.Module:
        model_config = self.config.get('model', {})
        model = create_integrated_model(
            self.config,
            yolo_cfg=model_config.get('yolo_cfg', 'yolov5l.yaml'),
            pretrained_yolo=model_config.get('pretrained_yolo', None)
        )
        model = model.to(self.device)
        return model
    
    def _create_optimizer(self) -> optim.Optimizer:
        train_config = self.config.get('training', {})
        lr = train_config.get('learning_rate', 1e-2)
        weight_decay = train_config.get('weight_decay', 5e-4)
        optimizer_type = train_config.get('optimizer', 'sgd').lower()
        
        if optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
            LOGGER.info(f'Optimizer: AdamW with lr={lr}, weight_decay={weight_decay}')
        else:
            momentum = train_config.get('momentum', 0.937)
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
            LOGGER.info(f'Optimizer: SGD with momentum={momentum}, weight_decay={weight_decay}')
        return optimizer
    
    def _create_scheduler(self):
        train_config = self.config.get('training', {})
        if not train_config.get('use_scheduler', True):
            return None
        epochs = train_config['epochs']
        def lf(x):
            return (1 - x / epochs) * 0.9 + 0.1
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        return scheduler
    
    # --- TensorBoard可視化メソッド（新規追加） ---
    def log_images_to_tensorboard(self, batch, outputs, epoch):
        """
        学習中の画像をTensorBoardに送信する
        - RGB / Thermal 原画像
        - Fusion結果（BBox付き）
        - 重みマップ（分布と画像）
        """
        if not self.tb_writer:
            return

        try:
            # バッチの先頭の1枚だけを使用
            idx = 0

            # 1. 重みのヒストグラム（分布）
            if 'rgb_weight' in outputs and 'ir_weight' in outputs:
                self.tb_writer.add_histogram('Weights/RGB_Distribution', outputs['rgb_weight'], epoch)
                self.tb_writer.add_histogram('Weights/Thermal_Distribution', outputs['ir_weight'], epoch)

                # 重みマップの画像化 (ヒートマップ風にグレースケールで)
                # (B, 1, H, W) -> (1, H, W)
                rgb_w_map = outputs['rgb_weight'][idx].detach().cpu()
                ir_w_map = outputs['ir_weight'][idx].detach().cpu()

                # TensorBoardは (C, H, W) を期待するが、重みは0-1なのでそのまま画像として扱える
                self.tb_writer.add_image('Fusion/Weight_Map_RGB', rgb_w_map, epoch)
                self.tb_writer.add_image('Fusion/Weight_Map_Thermal', ir_w_map, epoch)

            # 2. 元画像の復元 (Normalizeされているので戻す)
            # RGB: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            rgb_tensor = batch['rgb_image'][idx].detach().cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            rgb_vis = rgb_tensor * std + mean
            rgb_vis = torch.clamp(rgb_vis, 0, 1)
            self.tb_writer.add_image('Input/RGB', rgb_vis, epoch)

            # Thermal: mean=[0.5], std=[0.5]
            ir_tensor = batch['ir_image'][idx].detach().cpu()
            ir_vis = ir_tensor * 0.5 + 0.5
            ir_vis = torch.clamp(ir_vis, 0, 1)
            self.tb_writer.add_image('Input/Thermal', ir_vis, epoch)

            # 3. 融合画像と検出結果
            if 'yolo_input' in outputs:
                fused_tensor = outputs['yolo_input'][idx].detach().cpu() # (3, H, W) 0-1

                # OpenCVで描画するために Numpy (H, W, 3) に変換
                img_numpy = (fused_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
                # RGB -> BGR (OpenCV用)
                img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)

                # ターゲット（正解ラベル）の描画
                targets = batch['detection_gt'][idx]
                H, W = img_numpy.shape[:2]

                # 有効なターゲットのみ
                valid_mask = targets[:, 0] >= 0
                valid_targets = targets[valid_mask]

                for t in valid_targets:
                    cls, cx, cy, w, h = t.tolist()
                    # YOLO形式 (center_x, center_y, w, h) -> 左上・右下座標
                    x1 = int((cx - w/2) * W)
                    y1 = int((cy - h/2) * H)
                    x2 = int((cx + w/2) * W)
                    y2 = int((cy + h/2) * H)

                    # 緑の枠を描画
                    cv2.rectangle(img_numpy, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # クラス名描画（オプション）
                    label = self.class_names[int(cls)] if int(cls) < len(self.class_names) else str(int(cls))
                    cv2.putText(img_numpy, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # TensorBoard用に再度 Tensor (C, H, W) RGB に戻す
                img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)
                fused_vis = torch.from_numpy(img_numpy).permute(2, 0, 1)

                self.tb_writer.add_image('Fusion/Result_with_BBox', fused_vis, epoch)

        except Exception as e:
            LOGGER.warning(f"Failed to log images to TensorBoard: {e}")

    def _save_epoch_visualizations(self, epoch):
        """
        10エポックごとに検証データから5サンプルの可視化画像を保存
        """
        try:
            model = self.ema.ema if self.ema else self.model
            model.eval()

            # 検証データローダーから最初のバッチを取得
            val_iter = iter(self.val_loader)
            batch = next(val_iter)

            if batch is None:
                LOGGER.warning(f"Could not get validation batch for visualization at epoch {epoch}")
                return

            ir = batch['ir_image'].to(self.device)
            rgb = batch['rgb_image'].to(self.device)
            targets = batch['detection_gt'].to(self.device)

            # Forward pass to get fusion outputs
            with torch.no_grad():
                outputs = model(ir, rgb, targets, training=False)
                predictions = model.detect(ir, rgb)

            # Save visualization images
            self.save_visualization_images(batch, outputs, predictions, epoch)

            model.train()

        except Exception as e:
            LOGGER.warning(f"Failed to create epoch visualizations: {e}")

    def save_visualization_images(self, batch, outputs, predictions, epoch):
        """
        10エポックごとに5種類の画像をファイルとして保存
        - 融合画像にpred_boxとgt_boxが描画されたもの
        - RGB重みマップ
        - Thermal重みマップ
        """
        if epoch % 10 != 0:
            return

        try:
            # 保存先ディレクトリを作成
            vis_dir = self.save_dir / 'visualizations' / f'epoch_{epoch}'
            vis_dir.mkdir(parents=True, exist_ok=True)

            # バッチから最大5サンプルを取得
            num_samples = min(5, batch['ir_image'].size(0))

            for sample_idx in range(num_samples):
                # 1. 融合画像の取得と変換
                if 'yolo_input' not in outputs:
                    continue

                fused_tensor = outputs['yolo_input'][sample_idx].detach().cpu()  # (3, H, W) 0-1
                img_numpy = (fused_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
                img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
                H, W = img_numpy.shape[:2]

                # 2. GT Boxを緑色で描画
                targets = batch['detection_gt'][sample_idx]
                valid_mask = targets[:, 0] >= 0
                valid_targets = targets[valid_mask]

                for t in valid_targets:
                    cls, cx, cy, w, h = t.tolist()
                    x1 = int((cx - w/2) * W)
                    y1 = int((cy - h/2) * H)
                    x2 = int((cx + w/2) * W)
                    y2 = int((cy + h/2) * H)

                    # 緑色でGT Boxを描画
                    cv2.rectangle(img_numpy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"GT:{self.class_names[int(cls)]}" if int(cls) < len(self.class_names) else f"GT:{int(cls)}"
                    cv2.putText(img_numpy, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 3. Pred Boxを赤色で描画
                if predictions and sample_idx < len(predictions):
                    pred = predictions[sample_idx]
                    for p in pred:
                        if len(p) >= 6:
                            x1, y1, x2, y2, conf, cls = p[:6]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                            # 赤色でPred Boxを描画
                            cv2.rectangle(img_numpy, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            label = f"P:{self.class_names[int(cls)]} {conf:.2f}" if int(cls) < len(self.class_names) else f"P:{int(cls)} {conf:.2f}"
                            cv2.putText(img_numpy, label, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # 融合画像（BBox付き）を保存
                fusion_path = vis_dir / f'sample_{sample_idx}_fusion_boxes.png'
                cv2.imwrite(str(fusion_path), img_numpy)

                # 4. RGB重みマップを保存
                if 'rgb_weight' in outputs:
                    rgb_w_map = outputs['rgb_weight'][sample_idx].detach().cpu()  # (1, H, W)
                    rgb_w_np = (rgb_w_map.squeeze().numpy() * 255).astype(np.uint8)
                    # ヒートマップとして可視化
                    rgb_w_colored = cv2.applyColorMap(rgb_w_np, cv2.COLORMAP_JET)
                    rgb_w_path = vis_dir / f'sample_{sample_idx}_rgb_weight.png'
                    cv2.imwrite(str(rgb_w_path), rgb_w_colored)

                # 5. Thermal重みマップを保存
                if 'ir_weight' in outputs:
                    ir_w_map = outputs['ir_weight'][sample_idx].detach().cpu()  # (1, H, W)
                    ir_w_np = (ir_w_map.squeeze().numpy() * 255).astype(np.uint8)
                    # ヒートマップとして可視化
                    ir_w_colored = cv2.applyColorMap(ir_w_np, cv2.COLORMAP_JET)
                    ir_w_path = vis_dir / f'sample_{sample_idx}_thermal_weight.png'
                    cv2.imwrite(str(ir_w_path), ir_w_colored)

            LOGGER.info(f'Visualization images saved to {vis_dir}')

        except Exception as e:
            LOGGER.warning(f"Failed to save visualization images: {e}")

    def train(self):
        epochs = self.config['training']['epochs']
        LOGGER.info(f'\nStarting training for {epochs} epochs...')

        # 初期メモリ状態を表示
        if torch.cuda.is_available():
            LOGGER.info(f'GPU Memory Status:')
            LOGGER.info(f'  Total:     {torch.cuda.get_device_properties(0).total_memory / 1E9:.2f} GB')
            LOGGER.info(f'  Allocated: {torch.cuda.memory_allocated() / 1E9:.2f} GB')
            LOGGER.info(f'  Reserved:  {torch.cuda.memory_reserved() / 1E9:.2f} GB')

        if not self.results_file.exists():
             with open(self.results_file, 'w') as f:
                f.write('epoch,P,R,mAP@0.5,mAP@0.5:0.95,val_loss\n')

        if not self.per_class_results_file.exists():
            with open(self.per_class_results_file, 'w') as f:
                header = 'epoch,' + ','.join(self.class_names) + '\n'
                f.write(header)

        for epoch in range(self.start_epoch, epochs):
            self.current_epoch = epoch

            # Update epoch for MMDetection TOOD head
            if hasattr(self.model, 'mmdet_model') and hasattr(self.model.mmdet_model, 'bbox_head'):
                self.model.mmdet_model.bbox_head.epoch = epoch

            self.model.train()
            mem = '0.0G'
            mloss = torch.zeros(3, device=self.device)
            
            if RANK in {-1, 0}:
                pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                        bar_format='{l_bar}{bar:10}{r_bar}')
            else:
                pbar = enumerate(self.train_loader)
            
            self.optimizer.zero_grad()

            for i, batch in pbar:
                if batch is None:
                    continue
                
                time.sleep(0.05)

                try:
                    ir = batch['ir_image'].to(self.device, non_blocking=True)
                    rgb = batch['rgb_image'].to(self.device, non_blocking=True)
                    targets = batch['detection_gt'].to(self.device, non_blocking=True)
                except Exception as e:
                    LOGGER.error(f"Batch error: {e}")
                    raise
                
                # Forward
                outputs = self.model(ir, rgb, targets, training=True)
                losses = outputs['losses']
                loss = losses.get('total_loss', torch.tensor(0.0, device=self.device))

                # 融合損失の詳細を記録（TensorBoard用）
                self._last_fusion_losses = {
                    'ssim': outputs.get('ssim_loss', torch.tensor(0.0)).item() if torch.is_tensor(outputs.get('ssim_loss')) else 0,
                    'gradient': outputs.get('gradient_loss', torch.tensor(0.0)).item() if torch.is_tensor(outputs.get('gradient_loss')) else 0,
                    'entropy': outputs.get('entropy_loss', torch.tensor(0.0)).item() if torch.is_tensor(outputs.get('entropy_loss')) else 0,
                }

                loss = loss / self.accumulation_steps
                loss.backward()

                if (i + 1) % self.accumulation_steps == 0:
                    # Gradient Clipping（勾配爆発防止）
                    max_grad_norm = self.config.get('training', {}).get('max_grad_norm', 10.0)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # より積極的にキャッシュをクリア（メモリ不足対策）
                    if i % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                if self.ema:
                    self.ema.update(self.model)
                
                loss_items = self._get_loss_items(outputs, losses)
                mloss = (mloss * i + loss_items) / (i + 1)
                
                if RANK in {-1, 0}:
                    # 実際に割り当てられているメモリを表示（予約メモリではなく）
                    if torch.cuda.is_available():
                        mem_allocated = torch.cuda.memory_allocated() / 1E9
                        mem_reserved = torch.cuda.memory_reserved() / 1E9
                        mem = f'{mem_allocated:.1f}G'
                    else:
                        mem = '0.0G'
                    pbar.set_description(('%10s' * 2 + '%10.4g' * 3) % (f'{epoch + 1}/{epochs}', mem, *mloss))
                
                # ✅ ここでTensorBoardへの画像送信を実行 (各エポックの最初のバッチだけ)
                # メモリ節約のため10エポックごとに実行
                if i == 0 and RANK in {-1, 0} and epoch % 10 == 0:
                    self.log_images_to_tensorboard(batch, outputs, epoch)
                    # 可視化後にキャッシュをクリア
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if self.scheduler:
                self.scheduler.step()
            
            if RANK in {-1, 0}:
                results = self.validate()
                mp, mr, map50, map_50_95, val_loss, class_ap_dict = results

                self.results.append((epoch, [mp, mr, map50, map_50_95, val_loss]))
                self.per_class_results.append((epoch, class_ap_dict))

                # 10エポックごとに可視化画像を保存
                if epoch % 10 == 0:
                    self._save_epoch_visualizations(epoch)
                
                # ========== 毎エポックmAPを出力 ==========
                LOGGER.info('')
                LOGGER.info(f'Epoch {epoch + 1}/{epochs} Validation Results:')
                LOGGER.info(f'  Precision: {mp:.4f}  |  Recall: {mr:.4f}')
                LOGGER.info(f'  mAP@0.5:   {map50:.4f}  |  mAP@0.5:0.95: {map_50_95:.4f}')
                LOGGER.info(f'  Val Loss:  {val_loss:.4f}')
                
                # クラス別AP出力
                if class_ap_dict:
                    class_ap_str = '  Per-class AP@0.5: ' + ' | '.join(
                        [f'{name}: {class_ap_dict.get(name, 0.0):.4f}' for name in self.class_names]
                    )
                    LOGGER.info(class_ap_str)
                # ============================================
                
                if self.tb_writer:
                    self.tb_writer.add_scalar('train/box_loss', mloss[0], epoch)
                    self.tb_writer.add_scalar('train/cls_loss', mloss[1], epoch)
                    self.tb_writer.add_scalar('train/total_loss', mloss[2], epoch)

                    # 融合損失の詳細（最後のバッチの値を記録）
                    if hasattr(self, '_last_fusion_losses'):
                        self.tb_writer.add_scalar('fusion/ssim_loss', self._last_fusion_losses.get('ssim', 0), epoch)
                        self.tb_writer.add_scalar('fusion/gradient_loss', self._last_fusion_losses.get('gradient', 0), epoch)
                        self.tb_writer.add_scalar('fusion/entropy_loss', self._last_fusion_losses.get('entropy', 0), epoch)

                    self.tb_writer.add_scalar('metrics/precision', mp, epoch)
                    self.tb_writer.add_scalar('metrics/recall', mr, epoch)
                    self.tb_writer.add_scalar('metrics/mAP_0.5', map50, epoch)
                    self.tb_writer.add_scalar('metrics/mAP_0.5:0.95', map_50_95, epoch)
                    self.tb_writer.add_scalar('val/val_loss', val_loss, epoch)
                    
                    for j, x in enumerate(self.optimizer.param_groups):
                        self.tb_writer.add_scalar(f'lr/pg{j}', x['lr'], epoch)
                
                fi = fitness(np.array([mp, mr, map50, map_50_95, val_loss]).reshape(1, -1))
                
                # numpy配列をスカラーに変換
                fi_scalar = float(fi.item()) if isinstance(fi, np.ndarray) else float(fi)
                best_fitness_scalar = float(self.best_fitness.item()) if isinstance(self.best_fitness, np.ndarray) else float(self.best_fitness)
                
                # ベスト更新時のメッセージ
                if fi_scalar > best_fitness_scalar:
                    LOGGER.info(f'  ★ New Best! (fitness: {best_fitness_scalar:.4f} -> {fi_scalar:.4f})')
                    self.best_fitness = fi_scalar
                    self.save_checkpoint(self.best, epoch)
                else:
                    LOGGER.info(f'  Current best mAP@0.5: {best_fitness_scalar:.4f}')
                
                self.save_checkpoint(self.last, epoch) # 毎回保存
                
                with open(self.results_file, 'a') as f:
                    f.write(f'{epoch},{mp:.5f},{mr:.5f},'
                           f'{map50:.5f},{map_50_95:.5f},{val_loss:.5f}\n')
                
                with open(self.per_class_results_file, 'a') as f:
                    row = [str(epoch)]
                    for name in self.class_names:
                        row.append(f'{class_ap_dict.get(name, 0.0):.5f}')
                    f.write(','.join(row) + '\n')

                try:
                    npy_data = np.array([(x[0], *x[1]) for x in self.results])
                    np.save(self.results_npy_file, npy_data)
                    pc_data = []
                    for ep, d in self.per_class_results:
                        row = [ep]
                        for name in self.class_names:
                            row.append(d.get(name, 0.0))
                        pc_data.append(row)
                    np.save(self.per_class_npy_file, np.array(pc_data))
                except Exception as e:
                    LOGGER.warning(f"Failed to save binary results: {e}")
        
        if RANK in {-1, 0}:
            LOGGER.info(f'\nTraining complete ({epochs} epochs)')
            self.plot_results()
            LOGGER.info(f"\nResults saved to {self.save_dir}")
            LOGGER.info(f"Best fitness (mAP@0.5): {self.best_fitness:.3f}\n")
            if self.tb_writer:
                self.tb_writer.close()
    
    def _get_loss_items(self, outputs: Dict, losses: Dict) -> torch.Tensor:
        box_loss = 0.0
        cls_loss = 0.0
        total_loss = losses.get('total_loss', torch.tensor(0.0))

        if isinstance(losses, dict):
            try:
                if 'loss_bbox' in losses:
                    l = losses['loss_bbox']
                    if isinstance(l, (list, tuple)):
                         box_loss = sum(x.mean().item() for x in l if isinstance(x, torch.Tensor))
                    else:
                         box_loss = l.item() if torch.is_tensor(l) else float(l)
                if 'loss_cls' in losses:
                    l = losses['loss_cls']
                    if isinstance(l, (list, tuple)):
                         cls_loss = sum(x.mean().item() for x in l if isinstance(x, torch.Tensor))
                    else:
                         cls_loss = l.item() if torch.is_tensor(l) else float(l)
            except Exception:
                pass

        try:
            total_val = total_loss.item() if torch.is_tensor(total_loss) else float(total_loss)
        except Exception:
            total_val = 0.0

        return torch.tensor([box_loss, cls_loss, total_val], device=self.device)
    
    def validate(self) -> Tuple:
        model = self.ema.ema if self.ema else self.model
        model.eval()
        
        iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
        niou = iouv.numel()
        stats = []
        
        pbar = tqdm(self.val_loader, desc='Validating', bar_format='{l_bar}{bar:10}{r_bar}')
        
        for batch in pbar:
            if batch is None: continue
            ir = batch['ir_image'].to(self.device)
            rgb = batch['rgb_image'].to(self.device)
            targets = batch['detection_gt'].to(self.device)
            B, _, H, W = ir.shape
            
            with torch.no_grad():
                predictions = model.detect(ir, rgb)
            
            for si in range(B):
                pred = predictions[si] if si < len(predictions) else torch.zeros((0, 6), device=self.device)
                target_batch = targets[si]
                valid_mask = target_batch[:, 0] >= 0
                target = target_batch[valid_mask]
                
                if len(target) == 0:
                    if len(pred) > 0:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool, device=self.device),
                                    pred[:, 4], pred[:, 5], torch.zeros(0, device=self.device)))
                    continue
                
                target_boxes = torch.zeros((len(target), 6), device=self.device)
                target_boxes[:, 0] = 0
                target_boxes[:, 1] = target[:, 0]
                
                x_center = target[:, 1] * W
                y_center = target[:, 2] * H
                width = target[:, 3] * W
                height = target[:, 4] * H
                
                target_boxes[:, 2] = x_center - width / 2
                target_boxes[:, 3] = y_center - height / 2
                target_boxes[:, 4] = x_center + width / 2
                target_boxes[:, 5] = y_center + height / 2
                
                if len(pred) == 0:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool, device=self.device),
                                torch.zeros(0, device=self.device),
                                torch.zeros(0, device=self.device),
                                target_boxes[:, 1]))
                    continue
                
                correct = torch.zeros(len(pred), niou, dtype=torch.bool, device=self.device)
                iou = box_iou(target_boxes[:, 2:6], pred[:, :4])
                target_cls = target_boxes[:, 1]
                pred_cls = pred[:, 5]
                correct_class = target_cls.unsqueeze(1) == pred_cls.unsqueeze(0)
                iou = iou * correct_class.float()
                
                for iou_idx, iou_thresh in enumerate(iouv):
                    matches_mask = iou >= iou_thresh
                    if matches_mask.any():
                        gt_indices, pred_indices = torch.where(matches_mask)
                        iou_values = iou[gt_indices, pred_indices]
                        sorted_indices = iou_values.argsort(descending=True)
                        gt_indices = gt_indices[sorted_indices]
                        pred_indices = pred_indices[sorted_indices]
                        matched_gt = set()
                        matched_pred = set()
                        for gt_i, pred_i in zip(gt_indices.tolist(), pred_indices.tolist()):
                            if gt_i not in matched_gt and pred_i not in matched_pred:
                                correct[pred_i, iou_idx] = True
                                matched_gt.add(gt_i)
                                matched_pred.add(pred_i)
                stats.append((correct, pred[:, 4], pred[:, 5], target_boxes[:, 1]))
        
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
        if len(stats) and stats[0].any():
            names_dict = {i: name for i, name in enumerate(self.class_names)}
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=self.save_dir, names=names_dict)
            ap50, ap = ap[:, 0], ap.mean(1)
            mp, mr, map50, map = float(p.mean()), float(r.mean()), float(ap50.mean()), float(ap.mean())
            class_ap_dict = {self.class_names[int(c)]: float(ap50[i]) for i, c in enumerate(ap_class)}
        else:
            mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0
            class_ap_dict = {}
        
        val_loss = 0.0
        loss_count = 0
        use_mmdet = self.config.get('model', {}).get('use_mmdet', False) or self.config.get('use_mmdet', False)
        if use_mmdet:
            for batch in self.val_loader:
                if batch is None: continue
                ir = batch['ir_image'].to(self.device)
                rgb = batch['rgb_image'].to(self.device)
                targets_batch = batch['detection_gt'].to(self.device)
                with torch.no_grad():
                    outputs = self.model(ir, rgb, targets_batch, training=True)
                    if 'losses' in outputs and 'detection_loss' in outputs['losses']:
                        val_loss += outputs['losses']['detection_loss'].item()
                        loss_count += 1
        val_loss = val_loss / max(loss_count, 1)
        model.train()
        return (mp, mr, map50, map, val_loss, class_ap_dict)
    
    def save_checkpoint(self, path: Path, epoch: int, results=None):
        ckpt = {
            'epoch': epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.model) if YOLO_UTILS_AVAILABLE else self.model).half(),
            'ema': deepcopy(self.ema.ema).half() if self.ema else None,
            'updates': self.ema.updates if self.ema else None,
            'optimizer': self.optimizer.state_dict(),
            'date': str(datetime.now()),
            'results': self.results,
            'per_class_results': self.per_class_results
        }
        if results: ckpt['results'] = results
        torch.save(ckpt, path)
        if path == self.best:
            LOGGER.info(f'New best mAP@0.5={float(self.best_fitness):.3f}')
    
    def load_checkpoint(self, path: str):
        LOGGER.info(f'Loading checkpoint from {path}...')
        ckpt = torch.load(path, map_location=self.device)
        model = de_parallel(self.model) if YOLO_UTILS_AVAILABLE else self.model
        state = ckpt['model'].float().state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
        model.load_state_dict(state, strict=False)
        if self.ema and ckpt.get('ema'):
            ema_state = ckpt['ema'].float().state_dict() if hasattr(ckpt['ema'], 'state_dict') else ckpt['ema']
            self.ema.ema.load_state_dict(ema_state)
            self.ema.updates = ckpt.get('updates', 0)
        if ckpt.get('optimizer'):
            self.optimizer.load_state_dict(ckpt['optimizer'])
        self.best_fitness = ckpt.get('best_fitness', 0.0)
        self.start_epoch = ckpt.get('epoch', -1) + 1
        if ckpt.get('results'): self.results = list(ckpt['results'])
        if ckpt.get('per_class_results'): self.per_class_results = list(ckpt['per_class_results'])
        if ckpt.get('results'): LOGGER.info(f'Restored {len(self.results)} previous epoch results from checkpoint')
        self.dataloaders = create_simple_m3fd_data_loaders(self.config)
        self.train_loader = self.dataloaders['train']
        self.val_loader = self.dataloaders['val']
        LOGGER.info(f'Resumed from epoch {self.start_epoch}')
    
    def plot_results(self):
        if not self.results: return
        try:
            csv_file = self.save_dir / 'results.csv'
            with open(csv_file, 'w') as f:
                f.write('epoch,P,R,mAP@0.5,mAP@0.5:0.95,val_loss\n')
                for epoch, metrics in self.results:
                    f.write(f'{epoch},{metrics[0]:.5f},{metrics[1]:.5f},{metrics[2]:.5f},{metrics[3]:.5f},{metrics[4]:.5f}\n')
            if YOLO_UTILS_AVAILABLE:
                try:
                    plot_results(file=csv_file, on_plot=None)
                    LOGGER.info(f'Results plotted to {self.save_dir / "results.png"}')
                    return
                except: pass
            self._plot_custom()
        except Exception as e:
            LOGGER.error(f'Plot failed: {e}')
    
    def _plot_custom(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Training Results', fontsize=16, fontweight='bold')
        epochs = [r[0] for r in self.results]
        metrics = np.array([r[1] for r in self.results])
        titles = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95', 'Val Loss']
        for i, (ax, title) in enumerate(zip(axes.flat, titles)):
            if i < len(metrics[0]):
                ax.plot(epochs, metrics[:, i], marker='o', linewidth=2)
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.set_xlabel('Epoch')
                ax.grid(True, alpha=0.3)
        axes.flat[-1].axis('off')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'results.png', dpi=200)
        plt.close()
        LOGGER.info(f'Results plotted to {self.save_dir / "results.png"}')

def main():
    parser = argparse.ArgumentParser(prog='train_fixed.py')
    parser.add_argument('--config', type=str, default='config_dataset.yaml', help='Path to config file')
    parser.add_argument('--weights', type=str, default='', help='Initial weights path')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='Resume training from last checkpoint')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='./runs/train', help='Save results to project/name')
    parser.add_argument('--name', default='exp', help='Save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='Existing project/name ok, do not increment')
    parser.add_argument('--add-graph', action='store_true', help='Add model graph to TensorBoard')
    
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if args.epochs: config['training']['epochs'] = args.epochs
    if args.batch_size: config['dataset']['batch_size'] = args.batch_size
    
    save_dir = Path(args.project) / args.name
    if save_dir.exists() and not args.exist_ok and not args.resume:
        for i in range(2, 10000):
            if not (Path(args.project) / f'{args.name}{i}').exists():
                save_dir = Path(args.project) / f'{args.name}{i}'
                break
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    trainer = IntegratedModelTrainer(config, str(save_dir), args.device, add_graph=args.add_graph)
    if args.resume:
        ckpt = args.resume if isinstance(args.resume, str) else str(trainer.last)
        if Path(ckpt).exists():
            trainer.load_checkpoint(ckpt)
        else:
            LOGGER.warning(f'Resume checkpoint not found: {ckpt}. Starting from scratch.')

    try:
        trainer.train()
    except KeyboardInterrupt:
        LOGGER.info('\nInterrupted')
    except Exception as e:
        LOGGER.error(f'\nError: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()