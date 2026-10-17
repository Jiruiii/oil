"""Load a fold's baseline Prithvi+DeepLabV3 as a frozen false-positive probe."""
import sys
from pathlib import Path

import numpy as np
import yaml

from .config import REPO_ROOT

MAIN_DIR = REPO_ROOT / "main"


def read_run(run_dir):
    """Return (args, weights_path) of a finished baseline run directory."""
    run_dir = Path(run_dir)
    args_path, weights = run_dir / "args.yaml", run_dir / "weights" / "best.pt"
    for p in (args_path, weights):
        if not p.is_file():
            raise FileNotFoundError(f"Probe run is missing {p}")
    args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    for key in ("architecture_cfg", "dataset"):
        if not isinstance(args.get(key), dict):
            raise ValueError(f"{args_path} lacks '{key}'")
    return args, weights


def probe_settings(args):
    """Normalization and class index exactly as the probe was trained, never re-estimated."""
    ds = args["dataset"]
    mean, std = list(ds.get("mean") or []), list(ds.get("std") or [])
    arch = args["architecture_cfg"]
    if len(mean) != len(std) or len(mean) != int(arch.get("in_channels", len(mean))):
        raise ValueError("Probe mean/std do not match in_channels")
    if min(std) <= 0:
        raise ValueError("Probe std must be positive")
    if int(ds.get("target_class_index", 0)) != 0:
        raise ValueError("HNM_SI assumes the oil class is index 0")
    return mean, std


class Probe:
    """Callable returning P(oil) maps for a normalized (B, C, H, W) numpy batch."""

    def __init__(self, model, device, amp):
        self.model, self.device, self.amp = model, device, amp

    def __call__(self, batch):
        import torch
        x = torch.from_numpy(np.ascontiguousarray(batch)).to(self.device)
        with torch.no_grad(), torch.autocast(self.device.type, enabled=self.amp):
            logits = self.model(x)["out"]
        return torch.softmax(logits.float(), dim=1)[:, 0].cpu().numpy()


def load_probe(run_dir, device, amp):
    import torch
    args, weights = read_run(run_dir)
    arch = dict(args["architecture_cfg"])
    ckpt = Path(arch.get("backbone_ckpt", ""))
    arch["backbone_ckpt"] = str(ckpt if ckpt.is_absolute() else (REPO_ROOT / ckpt).resolve())
    sys.path.insert(0, str(MAIN_DIR))
    try:
        from deeplab_adapter import DeepLabAdapter
        adapter = DeepLabAdapter(model_path=None, dataset_cfg=args["dataset"],
                                 architecture_cfg=arch)
    finally:
        sys.path.remove(str(MAIN_DIR))
    payload = torch.load(weights, map_location="cpu", weights_only=True)
    state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    # Strict: a silent partial load would turn the probe into a random network.
    adapter.model.load_state_dict(state, strict=True)
    model = adapter.model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return Probe(model, device, amp)
