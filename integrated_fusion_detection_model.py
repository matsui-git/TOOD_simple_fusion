"""
統合融合検出モデル - 完全独立版 (損失計算修正版)
fusion_model.pyに依存せず、必要な融合機能を内蔵
位置合わせなし、融合と物体検出をエンドツーエンドで学習

主な修正点:
1. YOLOv5の損失計算時の出力形式を修正
2. ターゲット変換の改善
3. 訓練モードと評価モードの明確な分離
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
import timm

# MMDetectionのインポート (v2.x 対応)
try:
    import mmdet
    from mmdet.apis import init_detector, inference_detector
    from mmdet.models import build_detector
    from mmcv import Config
    from mmcv.runner import load_checkpoint

    MMDET_AVAILABLE = True
    print(f"MMDetection version: {mmdet.__version__}")
except ImportError as e:
    MMDET_AVAILABLE = False
    import traceback
    traceback.print_exc()
    logging.getLogger(__name__).warning(f"MMDetection import failed: {e}")

# YOLOv5のパスを追加 (後方互換性のため残すが、MMDet使用時は使わない)
YOLO_PATH = Path(__file__).parent / 'yolov5'
if YOLO_PATH.exists():
    sys.path.insert(0, str(YOLO_PATH))
    try:
        from models.yolo import Model as YOLOModel
        from utils.loss import ComputeLoss
        from utils.general import non_max_suppression
        YOLO_AVAILABLE = True
    except ImportError:
        YOLO_AVAILABLE = False
else:
    YOLO_AVAILABLE = False

logger = logging.getLogger(__name__)

# デバッグ用フラグ
DEBUG_SHAPES = False

class SimpleFusionNetwork(nn.Module):
    """
    シンプルな融合ネットワーク
    IR画像(1ch) + RGB画像(3ch) -> 融合重みマップ -> 融合画像(1ch)
    融合損失: SSIM + Gradient + Entropy
    """
    def __init__(self, backbone_name: str = 'resnet34', freeze_weights: bool = False,
                 ssim_weight: float = 1.0, gradient_weight: float = 1.0, entropy_weight: float = 0.1):
        super().__init__()
        self.backbone_name = backbone_name
        self.freeze_weights = freeze_weights

        # 融合損失の重み
        self.ssim_weight = ssim_weight
        self.gradient_weight = gradient_weight
        self.entropy_weight = entropy_weight
        
        # バックボーン: 2チャンネル入力(IR gray + RGB gray)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            in_chans=2  # IR(1ch) + RGB_gray(1ch)
        )
        
        # 最初の畳み込み層を2チャンネル用に調整
        if hasattr(self.backbone, 'conv1'):
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                2, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            # 重みの初期化
            with torch.no_grad():
                if old_conv.weight.size(1) >= 2:
                    self.backbone.conv1.weight[:, :2, :, :] = old_conv.weight[:, :2, :, :].clone()
                else:
                    nn.init.kaiming_normal_(self.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        self.feature_channels = self.backbone.feature_info.channels()[-1]
        
        # 重み予測ヘッド
        self.weight_head = nn.Sequential(
            nn.Conv2d(self.feature_channels, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),
            nn.Softmax(dim=1)
        )
        
        # 重み固定の場合
        if self.freeze_weights:
            for param in self.parameters():
                param.requires_grad = False
    
    def rgb_to_gray(self, rgb: torch.Tensor) -> torch.Tensor:
        """RGB -> グレースケール変換"""
        if rgb.size(1) == 3:
            weights = torch.tensor([0.2126, 0.7152, 0.0722], device=rgb.device).view(1, 3, 1, 1)
            return torch.sum(rgb * weights, dim=1, keepdim=True)
        return rgb
    
    def forward(self, ir_image: torch.Tensor, rgb_image: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, _, H, W = ir_image.shape
        rgb_gray = self.rgb_to_gray(rgb_image)
        
        # 【修正】正規化を簡略化（毎回.min()を計算するのは高コスト）
        # データセット側で正規化済みの場合は単純にclampのみ
        # -1〜1の範囲なら0〜1に変換、それ以外はそのままclamp
        ir_norm = ir_image
        rgb_norm = rgb_gray
        
        # 値の範囲をチェック（最初の要素だけで判定して高速化）
        if ir_image.view(-1)[0].item() < -0.5:
            ir_norm = (ir_image + 1) / 2
        if rgb_gray.view(-1)[0].item() < -0.5:
            rgb_norm = (rgb_gray + 1) / 2
        
        ir_norm = torch.clamp(ir_norm, 0, 1)
        rgb_norm = torch.clamp(rgb_norm, 0, 1)
        
        combined_input = torch.cat([ir_norm, rgb_norm], dim=1)
        features = self.backbone(combined_input)
        final_features = features[-1]  # (B, 512, H/32, W/32) 程度の小さい特徴マップ
        
        if DEBUG_SHAPES:
            print(f"Backbone raw output (ch, H, W): {final_features.shape[1:]}")

        # 【修正】中間解像度でweight_headを適用（バランス型）
        # 修正前: 512ch x 416 x 416 → 約346MB
        # 修正後: 512ch x 52 x 52 → 約5.5MB（約63分の1、品質と効率のバランス）
        mid_h, mid_w = H // 8, W // 8  # 416 → 52
        final_features_mid = F.interpolate(final_features, size=(mid_h, mid_w), mode='bilinear', align_corners=True)
        
        if DEBUG_SHAPES:
            print(f"Features (mid resolution) (ch, H, W): {final_features_mid.shape[1:]}")
        
        weights_mid = self.weight_head(final_features_mid)  # (B, 2, H/8, W/8)
        
        if DEBUG_SHAPES:
            print(f"Weight head output (mid) (ch, H, W): {weights_mid.shape[1:]}")
        
        # 最後に2chだけをフル解像度にアップサンプリング
        weights = F.interpolate(weights_mid, size=(H, W), mode='bilinear', align_corners=True)
        
        if DEBUG_SHAPES:
            print(f"Weight map (upsampled) (ch, H, W): {weights.shape[1:]}")

        ir_weight, rgb_weight = weights[:, 0:1, :, :], weights[:, 1:2, :, :]
        fused_image = ir_weight * ir_norm + rgb_weight * rgb_norm

        # 融合損失: SSIM + Gradient + Entropy
        ssim_loss = self._calculate_ssim_loss(fused_image, ir_norm, rgb_norm)
        gradient_loss = self._calculate_gradient_loss(fused_image, ir_norm, rgb_norm)
        entropy_loss = self._calculate_entropy_loss(fused_image)

        # 重み付け合計（設定ファイルから読み込んだ係数を使用）
        # SSIM: 構造保存、Gradient: エッジ保存、Entropy: 情報量最大化（負値で最大化）
        fusion_quality_loss = (self.ssim_weight * ssim_loss +
                               self.gradient_weight * gradient_loss +
                               self.entropy_weight * entropy_loss)

        return {
            'fused_image': fused_image,
            'ir_weight': ir_weight,
            'rgb_weight': rgb_weight,
            'fusion_loss': fusion_quality_loss,
            'ssim_loss': ssim_loss,
            'gradient_loss': gradient_loss,
            'entropy_loss': entropy_loss
        }

    def _calculate_ssim_loss(self, fused: torch.Tensor, ir: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """
        SSIM Loss: 融合画像がIR/RGB両方の構造を保持しているか評価
        Loss = 1 - (SSIM(fused, IR) + SSIM(fused, RGB)) / 2
        """
        try:
            ssim_ir = self._ssim(fused, ir)
            ssim_rgb = self._ssim(fused, rgb)
            # SSIMは1が最良なので、1から引いて損失にする
            return 1.0 - (ssim_ir + ssim_rgb) / 2
        except:
            return torch.tensor(0.0, device=fused.device)

    def _ssim(self, x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """
        Structural Similarity Index (SSIM) の計算
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # ガウシアンウィンドウの作成
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32, device=x.device) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(0) * g.unsqueeze(1)
        window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, window_size, window_size)

        # パディング
        pad = window_size // 2

        # 平均
        mu_x = F.conv2d(x, window, padding=pad)
        mu_y = F.conv2d(y, window, padding=pad)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        # 分散・共分散
        sigma_x_sq = F.conv2d(x ** 2, window, padding=pad) - mu_x_sq
        sigma_y_sq = F.conv2d(y ** 2, window, padding=pad) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=pad) - mu_xy

        # SSIM
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

        return ssim_map.mean()

    def _calculate_gradient_loss(self, fused: torch.Tensor, ir: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """
        Gradient Loss: エッジ情報を最大限保存
        融合画像の勾配がIR/RGBの最大勾配に近づくようにする
        """
        try:
            # Sobelフィルタ
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=torch.float32, device=fused.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   dtype=torch.float32, device=fused.device).view(1, 1, 3, 3)

            # 各画像の勾配計算
            grad_fused_x = F.conv2d(fused, sobel_x, padding=1)
            grad_fused_y = F.conv2d(fused, sobel_y, padding=1)
            grad_fused = torch.sqrt(grad_fused_x ** 2 + grad_fused_y ** 2 + 1e-8)

            grad_ir_x = F.conv2d(ir, sobel_x, padding=1)
            grad_ir_y = F.conv2d(ir, sobel_y, padding=1)
            grad_ir = torch.sqrt(grad_ir_x ** 2 + grad_ir_y ** 2 + 1e-8)

            grad_rgb_x = F.conv2d(rgb, sobel_x, padding=1)
            grad_rgb_y = F.conv2d(rgb, sobel_y, padding=1)
            grad_rgb = torch.sqrt(grad_rgb_x ** 2 + grad_rgb_y ** 2 + 1e-8)

            # IR/RGBの最大勾配
            grad_max = torch.max(grad_ir, grad_rgb)

            # 融合画像の勾配が最大勾配に近づくようにする
            return F.l1_loss(grad_fused, grad_max)
        except:
            return torch.tensor(0.0, device=fused.device)

    def _calculate_entropy_loss(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Entropy Loss: 融合画像の情報量を最大化
        エントロピーを最大化したいので、負値を返す
        """
        try:
            # 画像を0-1にクランプ
            fused_clamped = torch.clamp(fused, 0, 1)

            # ヒストグラムを微分可能な方法で近似
            # ソフトヒストグラム: 各ビンへの寄与を連続的に計算
            num_bins = 256
            batch_size = fused_clamped.size(0)

            # バッチ全体で計算
            fused_flat = fused_clamped.view(batch_size, -1)  # (B, H*W)

            # 各ピクセル値のビンへの寄与を計算
            bin_centers = torch.linspace(0, 1, num_bins, device=fused.device)
            sigma = 1.0 / num_bins

            # (B, H*W, 1) - (1, 1, num_bins) -> (B, H*W, num_bins)
            diff = fused_flat.unsqueeze(-1) - bin_centers.view(1, 1, -1)
            weights = torch.exp(-0.5 * (diff / sigma) ** 2)

            # 正規化してヒストグラムを得る
            hist = weights.sum(dim=1)  # (B, num_bins)
            hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-10)

            # エントロピー計算
            entropy = -torch.sum(hist * torch.log(hist + 1e-10), dim=1)

            # エントロピーを最大化したいので負値を返す
            return -entropy.mean()
        except:
            return torch.tensor(0.0, device=fused.device)

class FusionToYOLOAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_rgb = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.to_rgb.weight.fill_(1.0)
    def forward(self, fused_gray: torch.Tensor) -> torch.Tensor:
        return self.to_rgb(fused_gray)

class IntegratedFusionDetectionModel(nn.Module):
    def __init__(self, config: Dict, yolo_cfg: str = 'yolov5s.yaml', pretrained_yolo: Optional[str] = None):
        super().__init__()
        self.config = config
        self.num_classes = config['dataset']['num_classes']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 【修正箇所1】modelセクションから正しく設定を読み込むように変更
        model_config = config.get('model', {})

        # 融合損失の重みを取得
        fusion_losses_config = model_config.get('fusion_losses', {})
        ssim_weight = fusion_losses_config.get('ssim_weight', 1.0)
        gradient_weight = fusion_losses_config.get('gradient_weight', 1.0)
        entropy_weight = fusion_losses_config.get('entropy_weight', 0.1)

        self.fusion_network = SimpleFusionNetwork(
            model_config.get('fusion_backbone', 'resnet34'),
            freeze_weights=model_config.get('freeze_fusion_weights', False),
            ssim_weight=ssim_weight,
            gradient_weight=gradient_weight,
            entropy_weight=entropy_weight
        )
        self.fusion_to_yolo = FusionToYOLOAdapter()

        logger.info(f"Fusion losses: SSIM={ssim_weight}, Gradient={gradient_weight}, Entropy={entropy_weight}")
        
        # デバッグ出力
        print(f"DEBUG: MMDET_AVAILABLE = {MMDET_AVAILABLE}")
        print(f"DEBUG: config['use_mmdet'] = {config.get('use_mmdet')}")
        
        # MMDetection (TOOD) の初期化
        # configの構造に対応: config['model']['use_mmdet'] または config['use_mmdet']
        self.use_mmdet = config.get('use_mmdet') or model_config.get('use_mmdet', False)
        
        print(f"DEBUG: Resolved use_mmdet = {self.use_mmdet}")

        if self.use_mmdet:
            if not MMDET_AVAILABLE:
                raise RuntimeError("MMDetection is not installed but 'use_mmdet' is True.")
            
            mmdet_config = config.get('mmdet_config') or model_config.get('mmdet_config')
            mmdet_checkpoint = config.get('mmdet_checkpoint') or model_config.get('mmdet_checkpoint')
            
            if not mmdet_config or not os.path.exists(mmdet_config):
                raise FileNotFoundError(f"MMDetection config not found: {mmdet_config}")
            
            logger.info(f"Initializing MMDetection model from {mmdet_config}")

            # MMDet 2.x: 設定ファイルを読み込み
            cfg = Config.fromfile(mmdet_config)

            # train_cfg / test_cfg の抽出と整理
            # モデル内のtrain_cfg/test_cfgを優先的に取得（bbox_head用の設定が含まれている）
            model_train_cfg = cfg.model.get('train_cfg')
            model_test_cfg = cfg.model.get('test_cfg')

            # トップレベルのtrain_cfg/test_cfg
            top_train_cfg = cfg.get('train_cfg')
            top_test_cfg = cfg.get('test_cfg')

            # モデル構築用にはモデル内の設定を使用
            train_cfg = model_train_cfg if model_train_cfg is not None else top_train_cfg
            test_cfg = model_test_cfg if model_test_cfg is not None else top_test_cfg

            # 重複回避のためmodel内からは削除
            if 'train_cfg' in cfg.model:
                cfg.model.pop('train_cfg')
            if 'test_cfg' in cfg.model:
                cfg.model.pop('test_cfg')

            # クラス数の上書き (データセットに合わせて変更)
            if 'bbox_head' in cfg.model:
                cfg.model.bbox_head.num_classes = self.num_classes

            # メモリ最適化: gradient_checkpointing
            if config.get('training', {}).get('gradient_checkpointing', False):
                if 'backbone' in cfg.model:
                    cfg.model.backbone.with_cp = True
                    logger.info("Gradient checkpointing enabled for backbone")

            # モデル構築: build_detector のシグネチャ互換性を考慮して呼び出す
            try:
                # 近年のMMDetでは train_cfg/test_cfg を引数で渡せる
                self.mmdet_model = build_detector(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)
            except TypeError:
                # 古い/互換性のない実装では train_cfg キーワードを受け取らない
                # その場合はモデルを先に構築し、必要なら属性として設定する
                self.mmdet_model = build_detector(cfg.model)
                try:
                    if train_cfg is not None:
                        self.mmdet_model.train_cfg = train_cfg
                    if test_cfg is not None:
                        self.mmdet_model.test_cfg = test_cfg
                except Exception:
                    # 属性設定に失敗しても先に進める
                    logger.debug('Failed to attach train/test cfg to mmdet model; continuing')

            self.mmdet_model.to(self.device)

            # チェックポイントのロード (形状不一致を考慮してロード)
            if mmdet_checkpoint:
                logger.info(f"Loading checkpoint from {mmdet_checkpoint}")
                try:
                    checkpoint = torch.load(mmdet_checkpoint, map_location='cpu')
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                    
                    # 形状が一致しないレイヤー（主にヘッドのクラス分類部分）を除外してロード
                    model_state_dict = self.mmdet_model.state_dict()
                    filtered_state_dict = {}
                    for k, v in state_dict.items():
                        if k in model_state_dict:
                            if v.shape == model_state_dict[k].shape:
                                filtered_state_dict[k] = v
                            else:
                                logger.warning(f"Skipping layer {k} due to shape mismatch: checkpoint {v.shape} != model {model_state_dict[k].shape}")
                        else:
                            pass # モデルに存在しないキーは無視
                            
                    self.mmdet_model.load_state_dict(filtered_state_dict, strict=False)
                    logger.info("Checkpoint loaded successfully (with filtered layers)")
                    
                except Exception as e:
                    logger.error(f"Failed to load checkpoint: {e}")
            
            # MMDetectionモデルのクラス数をチェック（確認用）
            if hasattr(self.mmdet_model, 'bbox_head'):
                if hasattr(self.mmdet_model.bbox_head, 'num_classes') and self.mmdet_model.bbox_head.num_classes != self.num_classes:
                    logger.warning(f"Model num_classes ({self.mmdet_model.bbox_head.num_classes}) != dataset num_classes ({self.num_classes})")
        else:
            # YOLOv5 (既存のロジック)
            if not YOLO_AVAILABLE:
                raise RuntimeError("YOLOv5 not available. Check 'yolov5' directory.")
                
            self.yolo_model = self._initialize_yolo_model(yolo_cfg, pretrained_yolo)
            self.yolo_model.to(self.device)
            self._move_yolo_tensors_to_device(self.device)
            self._attach_yolo_hyperparameters()
            
            try:
                self.yolo_loss = ComputeLoss(self.yolo_model)
                self._move_loss_tensors_to_device(self.device)
            except Exception as e:
                logger.error(f"ComputeLoss init failed: {e}")
                raise

        self.fusion_loss_weight = config.get('fusion_loss_weight', 0.1)
        self.detection_loss_weight = config.get('detection_loss_weight', 1.0)
        # デバッグ用フラグ（設定で有効化可能）
        self._debug_printed_losses = False

    def _initialize_yolo_model(self, yolo_cfg: str, pretrained_yolo: Optional[str]) -> nn.Module:
        cfg_path = Path(yolo_cfg)
        if not cfg_path.exists():
            cfg_path = YOLO_PATH / 'models' / yolo_cfg
        
        if not cfg_path.exists():
             raise FileNotFoundError(f"YOLOv5 config not found: {yolo_cfg}")

        yolo = YOLOModel(str(cfg_path), ch=3, nc=self.num_classes)
        
        if pretrained_yolo and os.path.exists(pretrained_yolo):
            logger.info(f"Loading pretrained YOLOv5 from {pretrained_yolo}")
            ckpt = torch.load(pretrained_yolo, map_location='cpu')
            state_dict = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt
            # クラス数不一致への対応
            if self.num_classes != 80:
                 state_dict = {k: v for k, v in state_dict.items() 
                               if not any(x in k for x in ['model.24', 'anchors'])}
            yolo.load_state_dict(state_dict, strict=False)
            
        return yolo

    def _attach_yolo_hyperparameters(self):
        # YOLOv5のデフォルトハイパーパラメータ
        self.yolo_model.hyp = {
            'box': 0.05, 'cls': 0.5, 'cls_pw': 1.0, 'obj': 1.0, 'obj_pw': 1.0,
            'fl_gamma': 0.0, 'anchor_t': 4.0, 'label_smoothing': 0.0
        }

    def _move_yolo_tensors_to_device(self, device):
        for m in self.yolo_model.modules():
            if hasattr(m, 'anchors') and isinstance(m.anchors, torch.Tensor): 
                m.anchors = m.anchors.to(device)
            if hasattr(m, 'anchor_grid') and isinstance(m.anchor_grid, torch.Tensor): 
                m.anchor_grid = m.anchor_grid.to(device)

    def _move_loss_tensors_to_device(self, device):
        for attr_name in ['anchors', 'anchor_grid', 'stride']:
            if hasattr(self.yolo_loss, attr_name):
                attr = getattr(self.yolo_loss, attr_name)
                if isinstance(attr, torch.Tensor): 
                    setattr(self.yolo_loss, attr_name, attr.to(device))

    def forward(self, ir_image: torch.Tensor, rgb_image: torch.Tensor, 
                detection_targets: Optional[torch.Tensor] = None,
                training: bool = True) -> Dict[str, any]:
        
        fusion_outputs = self.fusion_network(ir_image, rgb_image)
        # 融合画像 (B, 1, H, W) -> (B, 3, H, W)
        fused_rgb = torch.clamp(self.fusion_to_yolo(fusion_outputs['fused_image']), 0, 1)
        
        losses_dict = {}
        
        if self.use_mmdet:
            # MMDetection (TOOD) のフォワードパス (v2.x対応)
            if training and detection_targets is not None:
                self.mmdet_model.train()

                # MMDet 2.x: img_metas と gt_bboxes, gt_labels を準備
                img_metas = self._create_img_metas(fused_rgb)
                gt_bboxes, gt_labels = self._convert_targets_to_mmdet_format(detection_targets, fused_rgb.shape)

                # MMDet 2.x: forward_train() メソッドを使用
                mmdet_losses = self.mmdet_model.forward_train(
                    fused_rgb,
                    img_metas=img_metas,
                    gt_bboxes=gt_bboxes,
                    gt_labels=gt_labels
                )

                # 損失の合計（リスト形式も再帰的に合計）
                det_loss = 0.0
                for loss_value in mmdet_losses.values():
                    if isinstance(loss_value, torch.Tensor):
                        det_loss += loss_value.mean()
                    elif isinstance(loss_value, (list, tuple)):
                        det_loss += sum(l.mean() for l in loss_value if isinstance(l, torch.Tensor))

                losses_dict['detection_loss'] = det_loss

                # mmdet_losses の値を平均化して losses_dict にマージ
                losses_dict.update({k: v.mean() if isinstance(v, torch.Tensor) else v for k, v in mmdet_losses.items()})

                # デバッグ: 損失キー/値を一度だけ出力（設定で有効化可能）
                try:
                    debug_enabled = bool(self.config.get('debug', {}).get('print_mmdet_losses', False))
                except Exception:
                    debug_enabled = False

                if debug_enabled and not getattr(self, '_debug_printed_losses', False):
                    logger.info('MMDet losses keys and values (first batch):')
                    for k, v in losses_dict.items():
                        try:
                            if isinstance(v, torch.Tensor):
                                logger.info(f"  {k}: tensor shape={tuple(v.shape)} value={float(v.item()) if v.numel()==1 else 'tensor'}")
                            else:
                                logger.info(f"  {k}: {v}")
                        except Exception as e:
                            logger.info(f"  {k}: (error reading value) {e}")
                    self._debug_printed_losses = True

            else:
                # 推論時 (MMDet 2.x対応)
                self.mmdet_model.eval()
                # MMDet 2.x: img_metas を作成（推論用）
                img_metas = self._create_img_metas(fused_rgb)
                # MMDet 2.x: simple_test() メソッドを使用
                mmdet_results = self.mmdet_model.simple_test(fused_rgb, img_metas, rescale=False)
                return {
                    **fusion_outputs,
                    'fused_rgb': fused_rgb,
                    'mmdet_results': mmdet_results
                }
        else:
            # YOLOv5のフォワードパス
            yolo_input = fused_rgb
            if training and detection_targets is not None:
                self.yolo_model.train()
                yolo_outputs = self.yolo_model(yolo_input)
            else:
                self.yolo_model.eval()
                with torch.no_grad():
                    yolo_outputs = self.yolo_model(yolo_input)

            if training and detection_targets is not None:
                if YOLO_AVAILABLE:
                    yolo_loss = self._compute_yolo_loss(yolo_outputs, detection_targets, yolo_input.shape)
                    losses_dict['detection_loss'] = yolo_loss
                else:
                    losses_dict['detection_loss'] = torch.tensor(0.0, device=self.device)

        # 融合損失の追加
        if training and detection_targets is not None:
            losses_dict['fusion_loss'] = fusion_outputs['fusion_loss']
            losses_dict['total_loss'] = (
                self.fusion_loss_weight * losses_dict['fusion_loss'] + 
                self.detection_loss_weight * losses_dict['detection_loss']
            )

        return {
            **fusion_outputs,
            'yolo_input': fused_rgb, # 互換性のため
            'losses': losses_dict
        }

    def _create_img_metas(self, images: torch.Tensor) -> List[Dict]:
        """MMDetection用のimg_metasを作成"""
        B, C, H, W = images.shape
        img_metas = []
        for i in range(B):
            img_metas.append({
                'img_shape': (H, W, C),
                'ori_shape': (H, W, C),
                'pad_shape': (H, W, C),
                'scale_factor': 1.0,
                'flip': False,
                'filename': f'batch_{i}', # ダミー
                'batch_input_shape': (H, W)
            })
        return img_metas

    def _convert_targets_to_mmdet_format(self, targets: torch.Tensor, img_shape: Tuple[int]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        YOLOターゲット [B, max_obj, 5] (cls, x, y, w, h) normalized
        -> MMDetectionターゲット
           gt_bboxes: List[Tensor(N, 4)] (x1, y1, x2, y2) absolute
           gt_labels: List[Tensor(N,)] (class_idx)
        """
        B, C, H, W = img_shape
        gt_bboxes = []
        gt_labels = []

        for i in range(B):
            batch_targets = targets[i] # [max_obj, 5]
            # 有効なターゲットのみ (class >= 0)
            valid_mask = batch_targets[:, 0] >= 0
            valid_targets = batch_targets[valid_mask]

            if len(valid_targets) > 0:
                cls_ids = valid_targets[:, 0].long()
                bboxes_norm = valid_targets[:, 1:5] # x, y, w, h

                # xywh (norm) -> xyxy (abs)
                x_c = bboxes_norm[:, 0] * W
                y_c = bboxes_norm[:, 1] * H
                w = bboxes_norm[:, 2] * W
                h = bboxes_norm[:, 3] * H

                x1 = x_c - w / 2
                y1 = y_c - h / 2
                x2 = x_c + w / 2
                y2 = y_c + h / 2

                bboxes = torch.stack([x1, y1, x2, y2], dim=1)
                # クリップ
                bboxes[:, 0::2] = bboxes[:, 0::2].clamp(0, W)
                bboxes[:, 1::2] = bboxes[:, 1::2].clamp(0, H)

                gt_bboxes.append(bboxes)
                gt_labels.append(cls_ids)
            else:
                gt_bboxes.append(torch.zeros((0, 4), device=self.device))
                gt_labels.append(torch.zeros((0,), dtype=torch.long, device=self.device))

        return gt_bboxes, gt_labels

    def _compute_yolo_loss(self, yolo_predictions, targets, input_shape):
        """
        YOLOv5の損失計算
        
        Args:
            yolo_predictions: YOLOモデルの出力（訓練時は train_out）
            targets: バッチ化されたターゲット [batch_size, max_objects, 5] (class, x, y, w, h)
            input_shape: 入力画像のshape [batch_size, 3, H, W]
        """
        try:
            device = self.device
            batch_size = input_shape[0]
            
            # YOLOv5の出力形式を正規化
            # 訓練時: (inference_out, train_out) のタプル、またはtrain_outのリスト
            if isinstance(yolo_predictions, tuple) and len(yolo_predictions) == 2:
                # (inference_out, train_out) 形式 → train_outを使用
                train_out = yolo_predictions[1]
            elif isinstance(yolo_predictions, list):
                # すでにtrain_out形式
                train_out = yolo_predictions
            else:
                logger.warning(f"Unexpected YOLO output format: {type(yolo_predictions)}")
                # ✅ loss_itemsをリセット
                self.yolo_loss.loss_items = None
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            # train_outが空でないことを確認
            if not train_out or len(train_out) == 0:
                logger.warning("YOLO train_out is empty")
                # ✅ loss_itemsをリセット
                self.yolo_loss.loss_items = None
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            # ターゲットをYOLOv5形式に変換
            yolo_targets = self._convert_targets_to_yolo_format(targets, batch_size)
            
            if yolo_targets is None or len(yolo_targets) == 0:
                logger.debug("No valid targets for loss computation")
                # ✅ loss_itemsをリセット
                self.yolo_loss.loss_items = None
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            yolo_targets = yolo_targets.to(device)
            
            # ComputeLossの実行
            # 戻り値: (total_loss, loss_items) where loss_items = [box, obj, cls]
            loss, loss_items = self.yolo_loss(train_out, yolo_targets)
            
            # ✅ 修正: loss_itemsを保存して訓練ループで取得できるようにする
            self.yolo_loss.loss_items = loss_items
            
            # デバッグ情報
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss detected: {loss.item()}")
                self.yolo_loss.loss_items = None
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            return loss
            
        except Exception as e:
            logger.error(f"YOLOv5 loss computation failed: {e}")
            import traceback
            traceback.print_exc()
            # ✅ loss_itemsをリセット
            self.yolo_loss.loss_items = None
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _convert_targets_to_yolo_format(self, targets: torch.Tensor, batch_size: int) -> Optional[torch.Tensor]:
        """
        データセットのターゲット形式をYOLOv5の損失計算用形式に変換
        
        入力: [batch_size, max_objects, 5] (class, x_center_norm, y_center_norm, w_norm, h_norm)
              パディングは -1 で表される
        
        出力: [N, 6] (image_idx, class, x_center_norm, y_center_norm, w_norm, h_norm)
              N = 全バッチの有効なオブジェクト数の合計
        """
        try:
            device = targets.device
            valid_targets = []
            
            for batch_idx in range(batch_size):
                # このバッチの全ターゲット
                batch_targets = targets[batch_idx]  # [max_objects, 5]
                
                # 有効なターゲットのみ抽出（class >= 0）
                valid_mask = batch_targets[:, 0] >= 0
                valid_batch_targets = batch_targets[valid_mask]
                
                if len(valid_batch_targets) > 0:
                    # バッチインデックスを追加
                    img_idx = torch.full(
                        (len(valid_batch_targets), 1), 
                        batch_idx, 
                        dtype=torch.float32, 
                        device=device
                    )
                    # [image_idx, class, x, y, w, h] の形式に変換
                    formatted_targets = torch.cat([img_idx, valid_batch_targets], dim=1)
                    valid_targets.append(formatted_targets)
            
            if not valid_targets:
                return None
            
            # 全バッチのターゲットを結合
            return torch.cat(valid_targets, dim=0)
            
        except Exception as e:
            logger.error(f"Target conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def detect(self, ir, rgb, conf_thres=0.25, iou_thres=0.45):
        """推論用メソッド"""
        self.eval()
        with torch.no_grad():
            out = self.forward(ir, rgb, training=False)
            
            if self.use_mmdet:
                # MMDetectionの結果をYOLO形式に変換
                mmdet_results = out.get('mmdet_results', [])
                batch_size = ir.shape[0]
                predictions = []
                
                for i in range(batch_size):
                    if i < len(mmdet_results):
                        # MMDetectionのbbox結果をYOLO形式に変換
                        bboxes = mmdet_results[i]
                        try:
                            # 出力がクラスごとのリスト (List[np.ndarray]) の場合
                            if isinstance(bboxes, (list, tuple)):
                                per_class_preds = []
                                for cls_idx, arr in enumerate(bboxes):
                                    if arr is None:
                                        continue
                                    import numpy as _np
                                    if isinstance(arr, _np.ndarray) and arr.size > 0:
                                        arr_t = torch.from_numpy(arr).to(self.device)
                                        # arr shape: (N, 5) -> x1,y1,x2,y2,score
                                        n = arr_t.shape[0]
                                        pred = torch.zeros((n, 6), device=self.device)
                                        pred[:, :4] = arr_t[:, :4]
                                        pred[:, 4] = arr_t[:, 4]
                                        pred[:, 5] = float(cls_idx)
                                        per_class_preds.append(pred)
                                if per_class_preds:
                                    predictions.append(torch.cat(per_class_preds, dim=0))
                                else:
                                    predictions.append(torch.zeros((0, 6), device=self.device))
                            else:
                                # 出力が ndarray または Tensor の場合
                                import numpy as _np
                                if isinstance(bboxes, _np.ndarray):
                                    if bboxes.size == 0:
                                        predictions.append(torch.zeros((0, 6), device=self.device))
                                    else:
                                        arr_t = torch.from_numpy(bboxes).to(self.device)
                                        n = arr_t.shape[0]
                                        pred = torch.zeros((n, 6), device=self.device)
                                        pred[:, :4] = arr_t[:, :4]
                                        # スコアが5列目にある場合
                                        if arr_t.shape[1] >= 5:
                                            pred[:, 4] = arr_t[:, 4]
                                        pred[:, 5] = 0
                                        predictions.append(pred)
                                elif isinstance(bboxes, torch.Tensor):
                                    if bboxes.numel() == 0:
                                        predictions.append(torch.zeros((0, 6), device=self.device))
                                    else:
                                        n = bboxes.shape[0]
                                        pred = torch.zeros((n, 6), device=self.device)
                                        pred[:, :4] = bboxes[:, :4]
                                        if bboxes.shape[1] >= 5:
                                            pred[:, 4] = bboxes[:, 4]
                                        pred[:, 5] = 0
                                        predictions.append(pred)
                                else:
                                    # 予期しない形式
                                    predictions.append(torch.zeros((0, 6), device=self.device))
                        except Exception as e:
                            logger.warning(f"Failed to parse mmdet result for image {i}: {e}")
                            predictions.append(torch.zeros((0, 6), device=self.device))
                    else:
                        predictions.append(torch.zeros((0, 6), device=self.device))
                
                return predictions
            else:
                # YOLOv5の推論出力から予測を取得
                if isinstance(out['yolo_outputs'], tuple):
                    preds = out['yolo_outputs'][0]  # (inference_out, _) の場合
                else:
                    preds = out['yolo_outputs']
                
                return non_max_suppression(preds, conf_thres, iou_thres) if YOLO_AVAILABLE else []

def create_integrated_model(config, yolo_cfg='yolov5s.yaml', pretrained_yolo=None):
    return IntegratedFusionDetectionModel(config, yolo_cfg, pretrained_yolo)

if __name__ == "__main__":
    # テスト実行時のみデバッグフラグを有効化
    DEBUG_SHAPES = True
    print("Running test mode...")
    
    # ダミー入力 (1, 1, 640, 640) and (1, 3, 640, 640)
    ir_dummy = torch.randn(1, 1, 640, 640)
    rgb_dummy = torch.randn(1, 3, 640, 640)
    
    print(f"Input IR shape: {ir_dummy.shape}")
    print(f"Input RGB shape: {rgb_dummy.shape}")
    
    model = SimpleFusionNetwork()
    model.eval()
    
    print("\n--- Forward Pass Shapes ---")
    with torch.no_grad():
        output = model(ir_dummy, rgb_dummy)
    print("---------------------------\n")