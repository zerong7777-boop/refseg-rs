from __future__ import annotations

import json
import os
from typing import Any, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from PIL import Image

from ..dataset import pil_to_tensor
from ..checkpoint import infer_checkpoint_architecture, resolve_runtime_checkpoint_path
from ..metrics import evaluate_predictions
from ..models import NativeXeMapRefSeg, PortableGroundingBertTextEncoder
from ..preprocess import GroundingImagePreprocessor
from .portable_swin import build_portable_swin_feature_extractor

DEFAULT_OPTIONAL_SITE_PACKAGES = ""
DEFAULT_VISUAL_BACKEND = "portable_swin"
DEFAULT_TEXT_BACKEND = "hf_bert"
DEFAULT_TEXT_MODEL_NAME = "bert-base-uncased"
DEFAULT_MODEL_EMBED_DIMS = 256
DEFAULT_BACKBONE_EMBED_DIMS = 128


def _select_device(raw_device: str) -> torch.device:
    if raw_device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(raw_device)
    return torch.device("cpu")


def _normalize_text_prompt(text: str) -> str:
    text = text.lower().strip()
    if not text.endswith("."):
        text = text + "."
    return text


def _resolve_architecture(
    checkpoint_path: str,
    embed_dims: int,
    backbone_embed_dims: int | None,
    neck_out_channels: int | None,
    infer_arch_from_checkpoint: bool,
) -> Dict[str, int | str]:
    resolved_embed_dims = int(embed_dims)
    resolved_backbone_embed_dims = (
        DEFAULT_BACKBONE_EMBED_DIMS if backbone_embed_dims is None else int(backbone_embed_dims)
    )
    resolved_neck_out_channels = int(neck_out_channels) if neck_out_channels is not None else resolved_embed_dims
    arch_source = "requested"

    if infer_arch_from_checkpoint and checkpoint_path:
        arch = infer_checkpoint_architecture(checkpoint_path)
        if arch.model_embed_dims is not None:
            resolved_embed_dims = int(arch.model_embed_dims)
            arch_source = "checkpoint"
        if arch.backbone_embed_dims is not None:
            resolved_backbone_embed_dims = int(arch.backbone_embed_dims)
            arch_source = "checkpoint"
        if arch.neck_out_channels is not None:
            resolved_neck_out_channels = int(arch.neck_out_channels)
            arch_source = "checkpoint"

    return {
        "embed_dims": resolved_embed_dims,
        "backbone_embed_dims": resolved_backbone_embed_dims,
        "neck_out_channels": resolved_neck_out_channels,
        "arch_source": arch_source,
    }


def _build_visual_extractor(
    visual_backend: str,
    optional_site_packages: str,
    backbone_embed_dims: int,
    neck_out_channels: int,
):
    if visual_backend != "portable_swin":
        raise ValueError(
            f"Unsupported visual backend: {visual_backend}. "
            "refseg_runtime now only supports visual_backend='portable_swin'."
        )
    return build_portable_swin_feature_extractor(
        optional_site_packages=optional_site_packages,
        backbone_embed_dims=backbone_embed_dims,
        neck_out_channels=neck_out_channels,
    )


def _freeze_module(module: torch.nn.Module | None) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = False


def _run_validation(
    model: NativeXeMapRefSeg,
    dataloader,
    device: torch.device,
    pred_threshold: float,
    gt_mode: str,
    iou_thresholds: list[float],
    max_batches: int = 0,
) -> Dict[str, Any]:
    model.eval()
    predictions = []
    gt_maps = []
    loss_values = []
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            texts = list(batch["text"])
            losses = model.loss(images, masks, texts=texts)
            loss_values.append(float(losses["loss_seg"].detach().cpu()))
            pred_maps = model.predict(images, texts=texts).detach().cpu().numpy().astype(np.float32)
            gt_batch = masks.squeeze(1).detach().cpu().numpy()
            predictions.extend(list(pred_maps))
            gt_maps.extend(list(gt_batch))
            if max_batches and (batch_idx + 1) >= max_batches:
                break
    p_at, miou, oiou = evaluate_predictions(
        predictions=predictions,
        gt_labels=gt_maps,
        iou_thresholds=iou_thresholds,
        gt_mode=gt_mode,
        pred_threshold=pred_threshold,
    )
    mean_loss = float(sum(loss_values) / len(loss_values)) if loss_values else 0.0
    return {
        "loss_seg": mean_loss,
        "miou": miou,
        "oiou": oiou,
        "p_at": p_at,
        "num_samples": len(gt_maps),
    }


def _is_better_metric(
    current_metrics: Dict[str, Any],
    best_metric_value: float | None,
    best_metric_name: str,
    min_delta: float = 0.0,
) -> tuple[bool, float]:
    current_value = float(current_metrics[best_metric_name])
    if best_metric_value is None:
        return True, current_value
    if best_metric_name == "loss_seg":
        return current_value < (best_metric_value - min_delta), current_value
    return current_value > (best_metric_value + min_delta), current_value


class NativeXeMapPredictor:
    def __init__(
        self,
        checkpoint_path: str = "",
        device: str = "cpu",
        embed_dims: int = DEFAULT_MODEL_EMBED_DIMS,
        backbone_embed_dims: int | None = None,
        neck_out_channels: int | None = None,
        infer_arch_from_checkpoint: bool = True,
        decoder_type: str = "unet",
        with_encoder: bool = True,
        optional_site_packages: str = DEFAULT_OPTIONAL_SITE_PACKAGES,
        visual_backend: str = DEFAULT_VISUAL_BACKEND,
        text_backend: str = DEFAULT_TEXT_BACKEND,
        text_model_name: str = DEFAULT_TEXT_MODEL_NAME,
        use_grounding_preprocess: bool = True,
        resize_to_original: bool = False,
        query_gate_cfg: Any = None,
    ) -> None:
        self.device = _select_device(device)
        self.image_preprocessor = GroundingImagePreprocessor() if use_grounding_preprocess else None
        self.resize_to_original = resize_to_original
        self.requested_checkpoint_path = checkpoint_path
        self.resolved_checkpoint_path = resolve_runtime_checkpoint_path(checkpoint_path)
        self.requested_embed_dims = int(embed_dims)
        self.requested_backbone_embed_dims = backbone_embed_dims
        self.requested_neck_out_channels = neck_out_channels
        self.infer_arch_from_checkpoint = bool(infer_arch_from_checkpoint)
        arch = _resolve_architecture(
            checkpoint_path=self.resolved_checkpoint_path,
            embed_dims=embed_dims,
            backbone_embed_dims=backbone_embed_dims,
            neck_out_channels=neck_out_channels,
            infer_arch_from_checkpoint=self.infer_arch_from_checkpoint,
        )
        self.embed_dims = int(arch["embed_dims"])
        self.backbone_embed_dims = int(arch["backbone_embed_dims"])
        self.neck_out_channels = int(arch["neck_out_channels"])
        self.arch_source = str(arch["arch_source"])
        visual_extractor = _build_visual_extractor(
            visual_backend=visual_backend,
            optional_site_packages=optional_site_packages,
            backbone_embed_dims=self.backbone_embed_dims,
            neck_out_channels=self.neck_out_channels,
        )
        text_encoder = None
        if text_backend == "hf_bert":
            text_encoder = PortableGroundingBertTextEncoder(
                name=text_model_name,
                output_dims=self.embed_dims,
                optional_site_packages=optional_site_packages,
            )
        self.model = NativeXeMapRefSeg(
            embed_dims=self.embed_dims,
            decoder_type=decoder_type,
            with_encoder=with_encoder,
            visual_extractor=visual_extractor,
            text_encoder=text_encoder,
            query_gate_cfg=query_gate_cfg,
        ).to(self.device)
        self.load_reports: Dict[str, Dict[str, Any]] = {}
        if checkpoint_path:
            self.load_partial_checkpoint(checkpoint_path)
        self.model.eval()

    def load_partial_checkpoint(self, checkpoint_path: str, strict: bool = False) -> Dict[str, Dict[str, Any]]:
        self.requested_checkpoint_path = checkpoint_path
        self.resolved_checkpoint_path = resolve_runtime_checkpoint_path(checkpoint_path)
        reports = self.model.load_partial_checkpoint(checkpoint_path, strict=strict)
        self.load_reports = reports
        return reports

    @torch.inference_mode()
    def predict(self, image_path: str, text: str) -> np.ndarray:
        normalized_text = _normalize_text_prompt(text)
        if self.image_preprocessor is not None:
            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(f"Unable to read image: {image_path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_tensor, meta = self.image_preprocessor(image_rgb)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            pred = self.model.predict(image_tensor, texts=[normalized_text])[0]
            if self.resize_to_original:
                pred = F.interpolate(
                    pred.unsqueeze(0).unsqueeze(0),
                    size=meta.original_size,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
        else:
            image = Image.open(image_path).convert("RGB")
            image_tensor = pil_to_tensor(image).unsqueeze(0).to(self.device)
            pred = self.model.predict(image_tensor, texts=[normalized_text])[0]
        return pred.detach().cpu().numpy().astype(np.float32)


def create_native_xemap_predictor(
    checkpoint_path: str = "",
    device: str = "cpu",
    embed_dims: int = DEFAULT_MODEL_EMBED_DIMS,
    backbone_embed_dims: int | None = None,
    neck_out_channels: int | None = None,
    infer_arch_from_checkpoint: bool = True,
    decoder_type: str = "unet",
    with_encoder: bool = True,
    optional_site_packages: str = DEFAULT_OPTIONAL_SITE_PACKAGES,
    visual_backend: str = DEFAULT_VISUAL_BACKEND,
    text_backend: str = DEFAULT_TEXT_BACKEND,
    text_model_name: str = DEFAULT_TEXT_MODEL_NAME,
    use_grounding_preprocess: bool = True,
    resize_to_original: bool = False,
    query_gate_cfg: Any = None,
    **_: Any,
) -> NativeXeMapPredictor:
    return NativeXeMapPredictor(
        checkpoint_path=checkpoint_path,
        device=device,
        embed_dims=embed_dims,
        backbone_embed_dims=backbone_embed_dims,
        neck_out_channels=neck_out_channels,
        infer_arch_from_checkpoint=infer_arch_from_checkpoint,
        decoder_type=decoder_type,
        with_encoder=with_encoder,
        optional_site_packages=optional_site_packages,
        visual_backend=visual_backend,
        text_backend=text_backend,
        text_model_name=text_model_name,
        use_grounding_preprocess=use_grounding_preprocess,
        resize_to_original=resize_to_original,
        query_gate_cfg=query_gate_cfg,
    )


def build_native_xemap_runtime(
    dataloader,
    work_dir: str,
    val_dataloader=None,
    checkpoint_path: str = "",
    device: str = "cpu",
    embed_dims: int = DEFAULT_MODEL_EMBED_DIMS,
    backbone_embed_dims: int | None = None,
    neck_out_channels: int | None = None,
    infer_arch_from_checkpoint: bool = True,
    decoder_type: str = "unet",
    with_encoder: bool = True,
    optional_site_packages: str = DEFAULT_OPTIONAL_SITE_PACKAGES,
    visual_backend: str = DEFAULT_VISUAL_BACKEND,
    text_backend: str = DEFAULT_TEXT_BACKEND,
    text_model_name: str = DEFAULT_TEXT_MODEL_NAME,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    optimizer_type: str = "adamw",
    max_steps: int = 10,
    max_epochs: int = 1,
    save_every: int = 0,
    grad_clip_norm: float = 0.0,
    use_amp: bool = False,
    freeze_backbone: bool = True,
    freeze_neck: bool = False,
    freeze_language_model: bool = True,
    freeze_encoder: bool = False,
    val_interval: int = 1,
    val_every_steps: int = 0,
    val_max_batches: int = 0,
    val_pred_threshold: float = 0.5,
    val_gt_mode: str = "gt_positive",
    val_iou_thresholds: str = "0.5,0.6,0.7,0.8,0.9",
    best_metric: str = "loss_seg",
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    log_every: int = 50,
    query_gate_cfg: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    device_obj = _select_device(device)
    requested_checkpoint_path = checkpoint_path
    resolved_checkpoint_path = resolve_runtime_checkpoint_path(checkpoint_path)
    parsed_val_iou_thresholds = [float(x.strip()) for x in val_iou_thresholds.split(",") if x.strip()]
    arch = _resolve_architecture(
        checkpoint_path=resolved_checkpoint_path,
        embed_dims=embed_dims,
        backbone_embed_dims=backbone_embed_dims,
        neck_out_channels=neck_out_channels,
        infer_arch_from_checkpoint=infer_arch_from_checkpoint,
    )
    resolved_embed_dims = int(arch["embed_dims"])
    resolved_backbone_embed_dims = int(arch["backbone_embed_dims"])
    resolved_neck_out_channels = int(arch["neck_out_channels"])
    arch_source = str(arch["arch_source"])
    visual_extractor = _build_visual_extractor(
        visual_backend=visual_backend,
        optional_site_packages=optional_site_packages,
        backbone_embed_dims=resolved_backbone_embed_dims,
        neck_out_channels=resolved_neck_out_channels,
    )
    text_encoder = None
    if text_backend == "hf_bert":
        text_encoder = PortableGroundingBertTextEncoder(
            name=text_model_name,
            output_dims=resolved_embed_dims,
            optional_site_packages=optional_site_packages,
        )
    model = NativeXeMapRefSeg(
        embed_dims=resolved_embed_dims,
        decoder_type=decoder_type,
        with_encoder=with_encoder,
        visual_extractor=visual_extractor,
        text_encoder=text_encoder,
        query_gate_cfg=query_gate_cfg,
    ).to(device_obj)
    load_reports: Dict[str, Any] = {}
    if checkpoint_path:
        predictor = NativeXeMapPredictor(
            checkpoint_path="",
            device=device,
            embed_dims=resolved_embed_dims,
            backbone_embed_dims=resolved_backbone_embed_dims,
            neck_out_channels=resolved_neck_out_channels,
            infer_arch_from_checkpoint=False,
            decoder_type=decoder_type,
            with_encoder=with_encoder,
            optional_site_packages=optional_site_packages,
            visual_backend=visual_backend,
            text_backend=text_backend,
            text_model_name=text_model_name,
            query_gate_cfg=query_gate_cfg,
        )
        predictor.model = model
        load_reports = predictor.load_partial_checkpoint(checkpoint_path, strict=False)

    if freeze_backbone:
        _freeze_module(getattr(model, "backbone", None))
    if freeze_neck:
        _freeze_module(getattr(model, "neck", None))
    if freeze_language_model:
        _freeze_module(getattr(model, "language_model", None))
    if freeze_encoder:
        _freeze_module(getattr(model, "encoder", None))
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    trainable_module_report: Dict[str, Dict[str, int]] = {}
    for module_name in (
        "backbone",
        "neck",
        "language_model",
        "encoder",
        "U_net_seg_head",
        "XeMap_seg_decoder",
    ):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        trainable_module_report[module_name] = {
            "total": int(sum(param.numel() for param in module.parameters())),
            "trainable": int(sum(param.numel() for param in module.parameters() if param.requires_grad)),
        }
    trainable_module_report["other"] = {
        "total": int(
            sum(
                param.numel()
                for name, param in model.named_parameters()
                if not any(name == key or name.startswith(f"{key}.") for key in trainable_module_report.keys())
            )
        ),
        "trainable": int(
            sum(
                param.numel()
                for name, param in model.named_parameters()
                if param.requires_grad
                and not any(name == key or name.startswith(f"{key}.") for key in trainable_module_report.keys())
            )
        ),
    }
    optimizer_kind = optimizer_type.lower().strip()
    if optimizer_kind == "adamw":
        optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_kind == "adam":
        optimizer = torch.optim.Adam(trainable_parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")
    os.makedirs(work_dir, exist_ok=True)
    amp_enabled = bool(use_amp and device_obj.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_metric_name = best_metric.strip().lower()
    if best_metric_name not in {"loss_seg", "miou", "oiou"}:
        raise ValueError(f"Unsupported best_metric: {best_metric}")
    early_stop_patience = max(0, int(early_stop_patience))
    early_stop_min_delta = max(0.0, float(early_stop_min_delta))

    def run_validation(epoch: int, step: int) -> Dict[str, Any]:
        metrics = _run_validation(
            model=model,
            dataloader=val_dataloader,
            device=device_obj,
            pred_threshold=val_pred_threshold,
            gt_mode=val_gt_mode,
            iou_thresholds=parsed_val_iou_thresholds,
            max_batches=val_max_batches,
        )
        metrics["epoch"] = epoch
        metrics["step"] = step
        return metrics

    def train() -> None:
        print(
            json.dumps(
                {
                    "event": "train_setup",
                    "freeze_backbone": freeze_backbone,
                    "freeze_neck": freeze_neck,
                    "freeze_language_model": freeze_language_model,
                    "freeze_encoder": freeze_encoder,
                    "trainable_parameter_count": int(sum(param.numel() for param in trainable_parameters)),
                    "trainable_modules": trainable_module_report,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        history = []
        validation_history = []
        step = 0
        best_metric_value = None
        best_checkpoint_path = ""
        last_checkpoint_path = ""
        stop_training = False
        validation_rounds_without_improvement = 0
        early_stop_triggered = False
        early_stop_state: Dict[str, Any] | None = None

        def handle_validation_metrics(metrics: Dict[str, Any], event_name: str) -> None:
            nonlocal best_metric_value
            nonlocal best_checkpoint_path
            nonlocal validation_rounds_without_improvement
            nonlocal stop_training
            nonlocal early_stop_triggered
            nonlocal early_stop_state

            validation_history.append(metrics)
            improved, current_metric_value = _is_better_metric(
                current_metrics=metrics,
                best_metric_value=best_metric_value,
                best_metric_name=best_metric_name,
                min_delta=early_stop_min_delta,
            )
            print(
                json.dumps(
                    {
                        "event": event_name,
                        "epoch": metrics["epoch"],
                        "step": metrics["step"],
                        "loss_seg": metrics["loss_seg"],
                        "miou": metrics["miou"],
                        "oiou": metrics["oiou"],
                        "num_samples": metrics["num_samples"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if improved:
                best_metric_value = current_metric_value
                best_checkpoint_path = os.path.join(work_dir, "best.pth")
                torch.save(model.state_dict(), best_checkpoint_path)
                validation_rounds_without_improvement = 0
                print(
                    json.dumps(
                        {
                            "event": "best_update",
                            "epoch": metrics["epoch"],
                            "step": metrics["step"],
                            "best_metric": best_metric_name,
                            "best_metric_value": best_metric_value,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            else:
                validation_rounds_without_improvement += 1
                if early_stop_patience > 0 and validation_rounds_without_improvement >= early_stop_patience:
                    stop_training = True
                    early_stop_triggered = True
                    early_stop_state = {
                        "metric": best_metric_name,
                        "patience": early_stop_patience,
                        "min_delta": early_stop_min_delta,
                        "rounds_without_improvement": validation_rounds_without_improvement,
                        "epoch": metrics["epoch"],
                        "step": metrics["step"],
                        "current_metric_value": current_metric_value,
                        "best_metric_value": best_metric_value,
                    }
                    print(
                        json.dumps(
                            {
                                "event": "early_stop",
                                **early_stop_state,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

        for epoch in range(1, max_epochs + 1):
            model.train()
            for batch in dataloader:
                images = batch["image"].to(device_obj)
                masks = batch["mask"].to(device_obj)
                texts = list(batch["text"])
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    losses = model.loss(images, masks, texts=texts)
                    loss_seg = losses["loss_seg"]
                scaler.scale(loss_seg).backward()
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(trainable_parameters, max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

                step += 1
                record = {"epoch": epoch, "step": step, "loss_seg": float(loss_seg.detach().cpu())}
                history.append(record)
                if log_every > 0 and (step == 1 or step % log_every == 0):
                    print(
                        json.dumps(
                            {
                                "event": "train_step",
                                "epoch": epoch,
                                "step": step,
                                "loss_seg": record["loss_seg"],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                if save_every and step % save_every == 0:
                    torch.save(model.state_dict(), os.path.join(work_dir, f"native_step_{step}.pth"))
                if val_dataloader is not None and val_every_steps > 0 and step % val_every_steps == 0:
                    metrics = run_validation(epoch=epoch, step=step)
                    handle_validation_metrics(metrics, event_name="val_step")
                    model.train()
                    if stop_training:
                        break
                if step >= max_steps:
                    stop_training = True
                    break

            last_checkpoint_path = os.path.join(work_dir, "last.pth")
            torch.save(model.state_dict(), last_checkpoint_path)

            should_run_epoch_val = (
                val_dataloader is not None
                and val_interval > 0
                and epoch % val_interval == 0
                and (val_every_steps <= 0 or step % val_every_steps != 0)
            )
            if should_run_epoch_val:
                metrics = run_validation(epoch=epoch, step=step)
                handle_validation_metrics(metrics, event_name="val_epoch")

            if stop_training:
                break

        with open(os.path.join(work_dir, "native_train_history.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "requested_checkpoint_path": requested_checkpoint_path,
                    "resolved_checkpoint_path": resolved_checkpoint_path,
                    "resolved_architecture": {
                        "embed_dims": resolved_embed_dims,
                        "backbone_embed_dims": resolved_backbone_embed_dims,
                        "neck_out_channels": resolved_neck_out_channels,
                        "source": arch_source,
                    },
                    "optimizer": {
                        "type": optimizer_kind,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "freeze_backbone": freeze_backbone,
                        "freeze_neck": freeze_neck,
                        "freeze_language_model": freeze_language_model,
                        "freeze_encoder": freeze_encoder,
                        "trainable_parameter_count": int(
                            sum(param.numel() for param in model.parameters() if param.requires_grad)
                        ),
                        "grad_clip_norm": grad_clip_norm,
                        "early_stop_patience": early_stop_patience,
                        "early_stop_min_delta": early_stop_min_delta,
                    },
                    "trainable_modules": trainable_module_report,
                    "last_checkpoint_path": last_checkpoint_path,
                    "best_checkpoint_path": best_checkpoint_path,
                    "best_validation_loss": best_metric_value if best_metric_name == "loss_seg" else None,
                    "best_metric_name": best_metric_name,
                    "best_metric_value": best_metric_value,
                    "best_checkpoint_rule": (
                        "lowest val loss_seg" if best_metric_name == "loss_seg" else f"highest val {best_metric_name}"
                    ),
                    "load_reports": load_reports,
                    "early_stop_triggered": early_stop_triggered,
                    "early_stop_state": early_stop_state,
                    "history": history,
                    "validation_history": validation_history,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    return {
        "model": model,
        "optimizer": optimizer,
        "train": train,
        "load_reports": load_reports,
        "requested_checkpoint_path": requested_checkpoint_path,
        "resolved_checkpoint_path": resolved_checkpoint_path,
        "resolved_architecture": {
            "embed_dims": resolved_embed_dims,
            "backbone_embed_dims": resolved_backbone_embed_dims,
            "neck_out_channels": resolved_neck_out_channels,
            "source": arch_source,
        },
    }
