import os

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch as t
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from jet_pytorch import Jet
from jet_pytorch.util import get_pretrained
from jet_pytorch.util import ImageDataset
from jet_pytorch.util import preprocess
from jet_pytorch.util import postprocess


def sample(
    images_path,
    jet_config={},
    device="cuda:0",
    checkpoint_path=None,
):
    t.set_float32_matmul_precision("medium")
    t.set_grad_enabled(False)

    dataset = ImageDataset(
        images_path,
        transform=v2.ToDtype(t.float32, scale=True),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=8,
        pin_memory=True,
        pin_memory_device=device,
        shuffle=True,
    )

    model = Jet(**jet_config)
    model = model.to(device)
    print(model)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        weights = t.load(checkpoint_path, map_location="cpu").get("model")
        if weights is None:
            raise RuntimeError("checkpoint missing 'model' key")

    else:
        weights = get_pretrained()

    model.load_state_dict(weights)
    model = t.compile(model)
    model.eval()

    for x in dataloader:
        x = preprocess(x).to(device)
        break

    x_r = Normal(0, 1).sample((16, 256, 48)).to(x.device)

    with t.autocast("cuda", dtype=t.bfloat16):
        emb, _ = model(x)
        x_recon, _ = model.inverse(emb)
        inv_r, _ = model.inverse(x_r)

    x = postprocess(x).cpu().numpy()
    emb = model.unpatchify(emb.float()).cpu().numpy()
    x_recon = postprocess(x_recon).cpu().numpy()
    inv_r = postprocess(inv_r).cpu().numpy()

    _, axs = plt.subplots(2, 2, figsize=(20, 20))
    plt.subplots_adjust(wspace=0.05, hspace=0.2)

    titles = ["original", "latent", "inverse", "noise inverse"]
    data = [x, emb, x_recon, inv_r]

    def create_grid(samples):
        h, w, c = samples[0].shape
        grid = np.zeros((h * 4, w * 4, c))
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                if idx < len(samples):
                    grid[i * h : (i + 1) * h, j * w : (j + 1) * w] = samples[idx]
        return grid

    for idx, (title, sample) in enumerate(zip(titles, data)):
        row, col = idx // 2, idx % 2
        grid = create_grid(sample)

        ax = axs[row, col]
        ax.imshow(np.clip(grid, 0, 1))
        ax.set_title(title, fontsize=20)
        ax.axis("off")

    os.makedirs("output", exist_ok=True)
    plt.tight_layout()
    plt.savefig("output/jet.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    fire.Fire(sample)
