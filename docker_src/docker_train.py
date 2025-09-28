import torchvision.io
from torchvision.io import read_image as original_read_image

# TODO Fix this properly
def safe_read_image(path):
    return original_read_image(str(path))

# Monkey patch global
import torchvision
torchvision.io.read_image = safe_read_image

import argparse
import os
from tqdm import tqdm
from jet_pytorch.train import train
from datasets import load_dataset
import wandb
from huggingface_hub import login


def save_hf_dataset_to_disk(hf_dataset, output_dir, percentage=0.01):
    os.makedirs(output_dir, exist_ok=True)
    subset = hf_dataset.select(range(int(len(hf_dataset) * percentage)))
    for i, sample in tqdm(enumerate(subset), total=len(subset)):    
        img = sample["image"]
        img = sample["image"].convert("RGB").resize((64, 64))
        img.save(os.path.join(output_dir, f"{i}.png"))


def get_hf_dataset():

    dataset = load_dataset(dataset_name)
    save_hf_dataset_to_disk(dataset["train"], f"./{dataset_name}_train")
    save_hf_dataset_to_disk(dataset["test"], f"./{dataset_name}_valid")


def train_jet():

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        entity=wandb_entity,
        config={
            "batch_size": 64,
            "accumulate_steps": 16,
            "epochs": 50,
            "learning_rate": 3e-4,
            "patch_size": 4,
            "patch_dim": 48,
            "n_patches": 256,
            "coupling_layers": 32,
        },
    )

    login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

    jet_config = dict(
        patch_size=4,
        patch_dim=48,
        n_patches=256,
        coupling_layers=32,
        block_depth=2,
        block_width=512,
        num_heads=8,
        scale_factor=2.0,
        coupling_types=(
            "channels", "channels",
            "channels", "channels",
            "spatial",
        ),
        spatial_coupling_projs=(
            "checkerboard", "checkerboard-inv",
            "vstripes", "vstripes-inv",
            "hstripes", "hstripes-inv",
        )
    )


    train(
        jet_config=jet_config,
        batch_size=32,
        accumulate_steps=16,
        device="cuda:0",
        epochs=250,
        warmup_percentage=0.1,
        max_grad_norm=1.0,
        learning_rate=3e-4,
        weight_decay=1e-5,
        adam_betas=(0.9, 0.95),
        images_path_train=f"./{dataset_name}_train",
        images_path_valid= f"./{dataset_name}_valid",
        num_workers=8,
        checkpoint_path="jet.pt",
        hf_repo_id=hf_repo_id
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_run_name", type=str)
    parser.add_argument("--hf_repo_id", type=str, default=None)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    wandb_entity = args.wandb_entity
    wandb_project = args.wandb_project
    wandb_run_name = args.wandb_run_name
    hf_repo_id = args.hf_repo_id

    get_hf_dataset()
    train_jet()
