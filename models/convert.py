import os
import argparse
from safetensors import safe_open
import torch


def convert_safetensors_to_pth(safetensors_path, pth_path=None):
    """
    Converts a .safetensors file to a .pth file.

    Args:
        safetensors_path (str): Path to the input .safetensors file.
        pth_path (str, optional): Path for the output .pth file.
                                  If not provided, it will be the same as the input but with .pth extension.
    """
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"The file {safetensors_path} does not exist.")

    if pth_path is None:
        pth_path = os.path.splitext(safetensors_path)[0] + ".pth"

    print(f"Loading tensors from {safetensors_path}...")
    state_dict = {}
    # 所有操作都在 with 块内完成，这是正确的
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    print(f"Saving state dict to {pth_path}...")
    torch.save(state_dict, pth_path)
    print("✅ Conversion successful!")


if __name__ == "__main__":
    WEIGHTS_PATH = "F:\\Model\\dino\\dinov3-vitl16-pretrain-lvd1689m\\model.safetensors"

    convert_safetensors_to_pth(WEIGHTS_PATH, None)