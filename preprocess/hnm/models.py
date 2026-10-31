"""Fixed-epoch miners; OOF held-out samples never select weights or epochs."""
import gc
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import WindowDataset


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(config, pretrained):
    import timm
    return timm.create_model(config["model_name"], pretrained=pretrained,
                             in_chans=len(config["mean"]), num_classes=2)


def train_miner(records, index, config, output, seed, device, dataset_cls=WindowDataset):
    seed_all(seed)
    model = create_model(config, pretrained=config["pretrained"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                           weight_decay=config["weight_decay"])
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    generator = torch.Generator().manual_seed(seed)
    pos = [r for r in records if r.label == 1]
    neg = [r for r in records if r.label == 0]
    if not pos or not neg:
        raise ValueError("Miner training needs both classes")
    rng = random.Random(seed)
    history = []
    for epoch in range(config["epochs"]):
        selected_neg = rng.sample(neg, min(len(neg), len(pos) * config["classifier_neg_ratio"]))
        samples = pos + selected_neg
        loader = DataLoader(dataset_cls(samples, index, config, augment=True),
                            batch_size=config["batch_size"], shuffle=True,
                            num_workers=config["workers"], generator=generator)
        model.train()
        total_loss = correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=amp):
                logits = model(x)
                ce = F.cross_entropy(logits, y, reduction="none", label_smoothing=.1)
                with torch.no_grad():
                    pt = torch.exp(-F.cross_entropy(logits, y, reduction="none"))
                alpha = torch.where(y == 1, .95, .05)
                loss = (alpha * (1 - pt) ** 1.5 * ce).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += float(loss.detach()) * len(y)
            correct += int((logits.detach().argmax(1) == y).sum())
            total += len(y)
        row = {"epoch": epoch + 1, "loss": total_loss / total,
               "accuracy": correct / total}
        history.append(row)
        print(f"{output.name} epoch {epoch + 1}/{config['epochs']}: {row}", flush=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, output)
    del model, opt, scaler
    gc.collect()
    if amp:
        torch.cuda.empty_cache()
    return history


def infer(records, index, config, checkpoint, device, embeddings=False):
    model = create_model(config, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    loader = DataLoader(WindowDataset(records, index, config),
                        batch_size=config["batch_size"], num_workers=config["workers"])
    arrays = []
    with torch.inference_mode():
        for x, _ in loader:
            x = x.to(device)
            if embeddings:
                feats = model.forward_features(x)
                value = model.forward_head(feats, pre_logits=True)
            else:
                value = (2 ** .5) * model(x).softmax(1)[:, 1]
            arrays.append(value.cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not arrays:
        raise ValueError("Empty inference pool")
    return np.concatenate(arrays)
