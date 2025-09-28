import os

import fire
import numpy as np
import torch as t
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
import wandb
from huggingface_hub import upload_file
from safetensors.torch import save_file

from jet_pytorch import Jet
from jet_pytorch.util import bits_per_dim
from jet_pytorch.util import cosine_schedule
from jet_pytorch.util import ImageDataset
from jet_pytorch.util import preprocess


def train(
    jet_config={},
    batch_size=64,
    accumulate_steps=16,
    epochs=50,
    warmup_percentage=0.1,
    max_grad_norm=1.0,
    learning_rate=3e-4,
    weight_decay=1e-5,
    adam_betas=(0.9, 0.95),
    images_path_train="/mnt/wsl/l/imagenet_downsampled",
    images_path_valid="/mnt/wsl/l/imagenet_downsampled",
    num_workers=16,
    device="cuda:0",
    checkpoint_path="jet_imagenet.pt",
    hf_repo_id = None
):
    t.set_float32_matmul_precision("medium")

    dataset = ImageDataset(
        images_path_train,
        transform=v2.ToDtype(t.float32, scale=True),
    )
    val_dataset = ImageDataset(
        images_path_valid,
        transform=v2.ToDtype(t.float32, scale=True),
    )

    dim_count = np.prod(dataset[0].shape)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        pin_memory_device=device,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    orig_model = Jet(**jet_config)
    orig_model = orig_model.to(device)
    print(orig_model)
    print(f"# Params: {sum(p.numel() for p in orig_model.parameters() if p.requires_grad):,}")
    model = t.compile(orig_model)

    opt = t.optim.AdamW(
        model.parameters(),
        betas=adam_betas,
        weight_decay=weight_decay,
        lr=learning_rate,
    )

    total_steps = (len(dataset) // (batch_size * accumulate_steps)) * epochs
    warmup_steps = int(total_steps * warmup_percentage)

    scheduler = cosine_schedule(
        opt,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_scale=0.01,
    )

    steps = 0
    log_bpd = 0.0
    log_nll = 0.0
    log_logdet = 0.0

    best_val_loss = float("inf")

    for epoch in range(epochs):
        pbar = tqdm(dataloader, total=len(dataloader) // accumulate_steps)

        for x in dataloader:
            x = preprocess(x).to(device)

            with t.autocast("cuda", dtype=t.bfloat16):
                logits, logdet = model(x)
                bpd, nll, logdet = bits_per_dim(logits, logdet, dim_count)

            bpd /= accumulate_steps
            bpd.backward()

            nll /= accumulate_steps
            logdet /= accumulate_steps
            log_bpd += bpd.detach().cpu().item()
            log_nll += nll.detach().cpu().item()
            log_logdet += logdet.detach().cpu().item()
            steps += 1

            if steps % accumulate_steps == 0:
                gn = clip_grad_norm_(model.parameters(), max_grad_norm).cpu().item()
                opt.step()
                scheduler.step()
                opt.zero_grad()
                pbar.update(1)
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(dict(
                    bpd=log_bpd,
                    nll=log_nll,
                    logdet=log_logdet,
                    gn=gn,
                    lr=lr,
                ))
                wandb.log({
                    "train/bpd": log_bpd,
                    "train/nll": log_nll,
                    "train/logdet": log_logdet,
                    "train/grad_norm": gn,
                    "train/lr": lr,
                    "epoch": epoch,
                })
                steps = 0
                log_bpd = 0.0
                log_nll = 0.0
                log_logdet = 0.0

        model.eval()

        t.save(
            dict(
                model=orig_model.state_dict(),
                opt=opt.state_dict(),
                epoch=epoch,
            ),
            checkpoint_path,
        )

        val_bpd = 0.0
        val_nll = 0.0
        val_logdet = 0.0

        pbar = tqdm(val_dataloader, total=len(val_dataloader))
        with t.no_grad():
            for x in val_dataloader:
                x = preprocess(x).to(device)

                with t.autocast("cuda", dtype=t.bfloat16):
                    logits, logdet = model(x)
                    bpd, nll, logdet = bits_per_dim(logits, logdet, dim_count)

                val_bpd += bpd.detach().cpu().item()
                val_nll += nll.detach().cpu().item()
                val_logdet += logdet.detach().cpu().item()
                pbar.update(1)

        val_bpd /= len(val_dataloader)
        val_nll /= len(val_dataloader)
        val_logdet /= len(val_dataloader)
        print(f"validation metrics - bpd: {val_bpd:.2f}, nll: {val_nll:.2f}, logdet: {val_logdet:.2f}")
        wandb.log({
            "val/bpd": val_bpd,
            "val/nll": val_nll,
            "val/logdet": val_logdet,
            "epoch": epoch,
        })

        # Save model if it's the best so far
        if val_bpd < best_val_loss:
            print(f"New best validation loss: {val_bpd:.4f} (prev {best_val_loss:.4f})")
            best_val_loss = val_bpd

            state_dict = orig_model.state_dict()
            save_file(state_dict, "jet_mnist.safetensors")

            upload_file(
                path_or_fileobj="jet_mnist.safetensors",
                path_in_repo="jet_mnist.safetensors",
                repo_id=hf_repo_id,
                repo_type="model"
            )

        model.train()


if __name__ == "__main__":
    fire.Fire(train)
