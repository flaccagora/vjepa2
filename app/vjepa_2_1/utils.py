# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys

import app.vjepa_2_1.models.predictor as vit_pred
import app.vjepa_2_1.models.vision_transformer as video_vit
import torch
import torch.nn.functional as F
import yaml
from app.vjepa_2_1.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import (
    CosineWDSchedule,
    LinearDecaySchedule,
    WarmupCosineSchedule,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def normalize_and_concat(tensor, embed_dim):
    """Split tensor into 4 chunks of size embed_dim along the last axis,
    apply LayerNorm to each chunk, then concatenate back."""
    chunks = [
        F.layer_norm(tensor[:, :, i * embed_dim : (i + 1) * embed_dim], (embed_dim,))
        for i in range(4)
    ]
    return torch.cat(chunks, dim=2)


def normalize_nested(nested, embed_dim):
    """Apply normalize_and_concat recursively over nested lists."""
    return [
        [[normalize_and_concat(z, embed_dim) for z in inner] for inner in outer]
        for outer in nested
    ]


def build_eval_args(
    model_name,
    patch_size,
    tubelet_size,
    num_frames,
    logging_folder,
    checkpoint,
    write_tag,
    eval_cfg_paths,
    uniform_power=False,
    use_sdpa=False,
    clip_duration=None,
    use_silu=False,
    wide_silu=True,
    tag=None,
):
    """
    Helper function to parse the pre-training configs to construct the
    evaluation configs, return as a list of eval configs.
    """
    import warnings

    if eval_cfg_paths is None:
        logger.info("No evaluations specified!")
        return

    eval_nodes = None
    eval_tasks_per_node = None
    args_eval = []
    for i, f in enumerate(eval_cfg_paths):
        with open(f, "r") as y_file:
            _args = yaml.load(y_file, Loader=yaml.FullLoader)
            _tag = _args.get("tag", "")
            _args["tag"] = f"{tag}-{_tag}"
            _nodes = _args.get("nodes", None)
            _tasks = _args.get("tasks_per_node", 8)
            eval_nodes = _nodes if eval_nodes is None else eval_nodes
            eval_tasks_per_node = (
                _tasks if eval_tasks_per_node is None else eval_tasks_per_node
            )
            if (eval_nodes != _nodes) or (eval_tasks_per_node != _tasks):
                warnings.warn(
                    "Configs for online evals must use same number of nodes for slurm-batch processing"
                )

            _args["pretrain"] = {}
            _args["pretrain"]["model_name"] = model_name
            _args["pretrain"]["patch_size"] = patch_size
            _args["pretrain"]["tubelet_size"] = tubelet_size
            _args["pretrain"]["uniform_power"] = uniform_power
            _args["pretrain"]["use_sdpa"] = use_sdpa
            _args["pretrain"]["clip_duration"] = clip_duration
            _args["pretrain"]["use_silu"] = use_silu
            _args["pretrain"]["wide_silu"] = wide_silu
            _args["pretrain"]["frames_per_clip"] = num_frames
            _args["pretrain"]["folder"] = logging_folder
            _args["pretrain"]["checkpoint"] = checkpoint
            _args["pretrain"]["write_tag"] = write_tag

            args_eval += [_args]

    return eval_nodes, eval_tasks_per_node, args_eval


def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
    is_anneal=False,
):
    logger.info(f"Loading {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = 0
    if not is_anneal:
        epoch = checkpoint.get("epoch", 0)

    def _strip_prefixes(key):
        for prefix in ("module.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if key.startswith("module.backbone."):
            key = key[len("module.backbone.") :]
        return key

    def _select_state_dict(*keys):
        for key in keys:
            if key in checkpoint:
                logger.info(f'using checkpoint key "{key}"')
                return checkpoint[key]
        return None

    def _load_state(model, pretrained_dict, label):
        if pretrained_dict is None:
            logger.info(f"No {label} state found in checkpoint; leaving initialized weights")
            return
        normalized_pretrained = {_strip_prefixes(k): v for k, v in pretrained_dict.items()}
        compatible_state = {}
        missing_keys = 0
        shape_mismatches = 0
        for k, v in model.state_dict().items():
            candidate = pretrained_dict.get(k)
            if candidate is None:
                candidate = normalized_pretrained.get(_strip_prefixes(k))
            if candidate is None:
                compatible_state[k] = v
                missing_keys += 1
            elif candidate.shape != v.shape:
                compatible_state[k] = v
                shape_mismatches += 1
            else:
                compatible_state[k] = candidate
        if missing_keys:
            logger.info(f"{label}: {missing_keys} keys were missing in loaded state dict")
        if shape_mismatches:
            logger.info(f"{label}: {shape_mismatches} keys had incompatible shapes and were reinitialized")
        msg = model.load_state_dict(compatible_state, strict=False)
        logger.info(f"loaded pretrained {label} from epoch {epoch} with msg: {msg}")

    encoder_state = _select_state_dict("encoder", "ema_encoder", "target_encoder")
    _load_state(encoder, encoder_state, "encoder")

    predictor_state = _select_state_dict("predictor")
    _load_state(predictor, predictor_state, "predictor")

    if target_encoder is not None:
        target_state = _select_state_dict("target_encoder", "ema_encoder", "encoder")
        _load_state(target_encoder, target_state, "target encoder")

    loaded_optimizer = False
    if "opt" in checkpoint:
        try:
            opt.load_state_dict(checkpoint["opt"])
            logger.info(f"loaded optimizer from epoch {epoch}")
            loaded_optimizer = True
        except ValueError:
            logger.info("Optimizer groups mismatch; reinitializing optimizer.")
    else:
        logger.info("No optimizer state found in checkpoint; starting a fresh optimizer")
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if not loaded_optimizer:
        epoch = 0
        logger.info("Starting fine-tuning from epoch 0 with a fresh optimizer schedule")
    logger.info(f"read-path: {r_path}")
    del checkpoint

    return (
        encoder,
        predictor,
        target_encoder,
        opt,
        scaler,
        epoch,
    )


def init_video_model(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=224,
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    uniform_power=False,
    use_mask_tokens=False,
    num_mask_tokens=2,
    zero_init_mask_tokens=True,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    is_causal=False,
    pred_is_causal=False,
    use_activation_checkpointing=False,
    return_all_tokens=False,
    chop_last_n_tokens=0,
    init_type="default",
    img_temporal_dim_size=None,
    n_registers=0,
    n_registers_predictor=0,
    has_cls_first=False,
    interpolate_rope=False,
    modality_embedding=False,
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        is_causal=is_causal,
        use_rope=use_rope,
        init_type=init_type,
        img_temporal_dim_size=img_temporal_dim_size,
        n_registers=n_registers,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        modality_embedding=modality_embedding,
    )
    encoder = MultiSeqWrapper(encoder)
    predictor = vit_pred.__dict__["vit_predictor"](
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=(
            encoder.backbone.num_heads if pred_num_heads is None else pred_num_heads
        ),
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        is_causal=pred_is_causal,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        return_all_tokens=return_all_tokens,
        chop_last_n_tokens=chop_last_n_tokens,
        n_registers=n_registers_predictor,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        modality_embedding=modality_embedding,
        img_temporal_dim_size=img_temporal_dim_size,
    )
    predictor = PredictorMultiSeqWrapper(predictor)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"Predictor number of parameters: {count_parameters(predictor)}")

    return encoder, predictor


def init_opt(
    is_anneal,
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    use_radamw=False,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
):
    param_groups = [
        {
            "params": (
                p
                for n, p in encoder.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in encoder.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]

    if use_radamw:
        from src.utils.adamw import AdamW as RAdamW

        logger.info("Using Rescaled-AdamW")
        optimizer = RAdamW(param_groups, betas=betas, eps=eps)
    else:
        logger.info("Using AdamW")
        optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)

    if not is_anneal:
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    else:
        scheduler = LinearDecaySchedule(
            optimizer,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )

    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler
