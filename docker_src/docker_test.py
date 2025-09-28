from torch.distributions import Normal
from jet_pytorch import Jet
from torchvision.utils import save_image
import os
from jet_pytorch.util import get_pretrained

# Config Jet model
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
weights = get_pretrained()
model.load_state_dict(weights)

# Generate latent vectors z ~ N(0, 1)
batch_size = 16
n_patches = 256
patch_dim = 48
z = Normal(0, 1).sample((batch_size, n_patches, patch_dim))

# Rebuild imgs from z
img, _ = model.inverse(z)  # img: (B, H, W, C)

# Reformat (B, C, H, W)
img = img.permute(0, 3, 1, 2)

# Normalize imgs [0,1]
img_min = img.amin(dim=(1, 2, 3), keepdim=True)
img_max = img.amax(dim=(1, 2, 3), keepdim=True)
img = (img - img_min) / (img_max - img_min + 1e-6)
img = img.clamp(0, 1)

# Save imgs
os.makedirs("output", exist_ok=True)
for i in range(img.size(0)):
    save_image(img[i], f"output/sample_{i:03d}.png")
save_image(img, "output/grid.png", nrow=4)
