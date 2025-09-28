# Jet PyTorch
![](./output/jet_readme.png)

This repo contains a PyTorch port of the code from [Jet: A Modern Transformer-Based Normalizing Flow](https://arxiv.org/abs/2412.15129) which was originally written in Jax and included as part of [big_vision](https://github.com/google-research/big_vision/blob/main/big_vision/models/proj/jet/jet.py).

A checkpoint pretrained on downsampled imagenet 64x64 is provided via [the hugging face hub](https://huggingface.co/btrude/jet_imagenet64x64_200m) as well as barebones single-GPU training and inference scripts.

## Installation
The code in this repo requires a CUDA-capable GPU which can perform operations at `bfloat16` precision.

```sh
pip install jet-pytorch
```

The models from the original paper/this reproduction are trained using downsampled imagenet which is available from a few different sources, including [Kaggle](https://www.kaggle.com/datasets/ayaroshevskiy/downsampled-imagenet-64x64). The `ImageDataset` provided in this repository does not load/support labels and does not require any specific directory structure.

## Usage
```py
from jet_pytorch import Jet

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
model = Jet(**jet_config)
```

This is the default Jet configuration and that which matches the pretrained weights available on the [huggingface hub](https://huggingface.co/btrude/jet_imagenet64x64_200m)

### Download and/or load pretrained imagenet 64x64 weights
```py
from jet_pytorch.util import get_pretrained

weights = get_pretrained()
model.load_state_dict(weights)
```

### Sample from a Jet
```py
from torch.distributions import Normal

batch_size = 16
n_patches = 256
patch_dim = 48
pdf = Normal(0, 1)
z = pdf.sample((batch_size, n_patches, patch_dim))
img, logdet = model.inverse(z)
```

### Training a Jet
```py
from jet_pytorch.train import train

jet_config = dict(...)
train(
    jet_config=jet_config,
    batch_size=64,
    accumulate_steps=16,
    device="cuda:0",
    epochs=50,
    warmup_percentage=0.1,
    max_grad_norm=1.0,
    learning_rate=3e-4,
    weight_decay=1e-5,
    adam_betas=(0.9, 0.95),
    images_path_train="/path/to/train/images",
    images_path_valid="/path/to/validation/images",
    num_workers=8,
    checkpoint_path="jet.pt",
)
```
The training code favors gradient accumulation over alternatives like gradient checkpointing, allowing this script to run on GPUs with less than 8GB of VRAM. The true batch size is thus equal to `batch_size * accumulate_steps`. Note that the default configuration assumes at least 24GB of VRAM.

### Create visualizations
```py
from jet_pytorch.sample import sample

# Creates visualization using the default Jet config/pretrained weights
sample("path/to/your/images")

# Creates visualization using default Jet config/a local checkpoint
sample(
    "path/to/your/images",
    checkpoint_path="path/to/your/checkpoint.pt",
)

# Creates visualization using custom Jet config/a local checkpoint
jet_config = dict(...)
sample(
    "path/to/your/images",
    jet_config=jet_config,
    checkpoint_path="path/to/your/checkpoint.pt",
)
```
Visualization results are stored at `output/jet.png`

### Run docker

Execute from ./jet-pytorch/

Available cards: `RTX4090`

#### Create the docker
```bash
docker build -f dockerfiles/your_card/Dockerfile -t jet .
```

#### Run the docker
```bash
docker run -it --gpus '"device=0"' -v $(pwd)/output:/app/output --ipc=host jet
```