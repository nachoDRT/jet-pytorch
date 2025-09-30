import math
import os
from functools import partial
from pathlib import Path

import torch as t
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.distributions import Normal
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from torchvision.io import read_image


def get_pretrained(repo_id: str = None, model_name:str = None):
    if model_name is None:
        pretrained_path = "models/jet_imagenet64x64_200m/jet_imagenet64x64_200m.safetensors"
    else:
        pretrained_path = f"models/{model_name}/{model_name}.safetensors"
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(pretrained_path):
        _ = hf_hub_download(
            repo_id=repo_id,
            filename=f"{model_name}.safetensors",
            local_dir=f"models/{model_name}",
        )

    return load_file(pretrained_path)


def _cosine_schedule(
    current_step,
    *,
    num_warmup_steps,
    num_training_steps,
    min_lr_scale,
    num_cycles,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(
        min_lr_scale,
        min_lr_scale + 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * (2.0) * progress)) * (1.0 - min_lr_scale),
    )


def cosine_schedule(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    min_lr_scale=0.1,
    num_cycles=0.5,
    last_epoch=-1,
):
    lr_lambda = partial(
        _cosine_schedule,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_scale=min_lr_scale,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        root = Path(root)
        self.files = [p for p in Path(root).rglob(f"*.png")]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = read_image(self.files[index])
        if self.transform is not None:
            image = self.transform(image)
        return image


def preprocess(x):
    x = x * 2 - 1  # 0,1 -> -1,1
    x = x + t.empty(x.shape).uniform_(0.0, 1 / 127.5)
    return x.permute(0, 2, 3, 1)


def postprocess(x):
    return (x + 1) / 2  # -1,1 -> 0,1


def bits_per_dim(logits, logdet, dim_count, pdf=None):
    if pdf is None:
        pdf = Normal(0.0, 1.0)

    nll = -pdf.log_prob(logits)
    nll = t.sum(
        nll + t.log(t.tensor(127.5, device=nll.device)),
        dim=tuple(range(1, nll.dim()))
    )
    nll = nll.unsqueeze(-1)

    bits = nll - logdet
    normalizer = t.log(t.tensor(2, device=nll.device)) * dim_count

    return (
        bits.mean() / normalizer,
        nll.mean() / normalizer,
        logdet.mean() / normalizer
    )
