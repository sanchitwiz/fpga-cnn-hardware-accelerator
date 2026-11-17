
import torch
import json

from pathlib import Path

from software.config import DEVICE
from software.dataset import get_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_PATH = (
    PROJECT_ROOT / "quantization" / "input_scale.json"
)


def calibrate_input_scale():

    _, val_loader, _ = get_dataloaders()

    max_value = 0.0

    print("Calibrating input activation range...")

    with torch.no_grad():

        for images, _ in val_loader:

            batch_max = images.abs().max().item()

            max_value = max(max_value, batch_max)

    scale = max(max_value / 127.0, 1e-8)

    return scale, max_value


def main():

    print("Using device:", DEVICE)

    scale, max_value = calibrate_input_scale()

    metadata = {
        "dtype": "int8",
        "quantization": "symmetric",
        "zero_point": 0,
        "max_absolute_value": max_value,
        "scale": scale
    }

    with open(OUTPUT_PATH, "w") as file:
        json.dump(metadata, file, indent=4)

    print("\n========== INPUT QUANTIZATION ==========")
    print(f"Maximum absolute input : {max_value:.6f}")
    print(f"Input scale            : {scale:.10f}")
    print(f"Input range            : {-127 * scale:.6f} to {127 * scale:.6f}")
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
