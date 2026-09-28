"""Minimal Orca-4B video hidden-state extractor."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch


def load_orca(checkpoint_dir: str | Path, device: torch.device):
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration
    root = Path(checkpoint_dir).expanduser().resolve()
    config_dir, weights = root / "vlm_config", root / "model.safetensors"
    if not config_dir.is_dir() or not weights.is_file():
        raise FileNotFoundError(f"Orca checkpoint requires vlm_config/ and model.safetensors: {root}")
    processor = AutoProcessor.from_pretrained(config_dir)
    config = AutoConfig.from_pretrained(config_dir)
    with torch.device("meta"):
        model = Qwen3_5ForConditionalGeneration(config)
    model.to_empty(device=device)
    raw = load_file(str(weights), device=str(device))
    state = {k[len("vlm.model."):]: v for k, v in raw.items() if k.startswith("vlm.model.")}
    if not state:
        raise RuntimeError("Orca checkpoint contains no vlm.model.* weights")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Orca weights mismatch: missing={incompatible.missing_keys[:5]} unexpected={incompatible.unexpected_keys[:5]}")
    model = model.to(dtype=torch.bfloat16).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, processor


def extract_video_feature(model, processor, frames: np.ndarray, prompt: str, device: torch.device):
    messages = [{"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors="pt", enable_thinking=False,
        processor_kwargs={"videos_kwargs": {"do_sample_frames": False}})
    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device=device, dtype=torch.bfloat16 if key == "pixel_values_videos" else None)
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    hidden = outputs.hidden_states[-1]
    mask = inputs["input_ids"].eq(int(model.config.video_token_id))
    if hidden.size(0) != 1 or not mask.any():
        raise RuntimeError(f"invalid Orca video tokens: hidden={tuple(hidden.shape)} count={int(mask.sum())}")
    tokens = hidden[0, mask[0]].float().cpu()
    return tokens.mean(0), int(tokens.shape[0])


def extract_video_tokens(model, processor, frames: np.ndarray, prompt: str, device: torch.device):
    """Return the complete last-layer video-token sequence for attentive readouts."""
    messages = [{"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors="pt", enable_thinking=False,
        processor_kwargs={"videos_kwargs": {"do_sample_frames": False}})
    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device=device, dtype=torch.bfloat16 if key == "pixel_values_videos" else None)
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    hidden = outputs.hidden_states[-1]
    mask = inputs["input_ids"].eq(int(model.config.video_token_id))
    if hidden.size(0) != 1 or not mask.any():
        raise RuntimeError(f"invalid Orca video tokens: hidden={tuple(hidden.shape)} count={int(mask.sum())}")
    tokens = hidden[0, mask[0]].float().cpu()
    return tokens, int(tokens.shape[0])
