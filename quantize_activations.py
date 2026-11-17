
import torch
import torch.nn as nn
import json

from pathlib import Path

from software.config import DEVICE, BEST_MODEL_PATH
from software.dataset import get_dataloaders
from software.model import CIFAR10_CNN


PROJECT_ROOT = Path(__file__).resolve().parent.parent

WEIGHTS_PATH = PROJECT_ROOT / "quantization" / "quantized_weights.pth"

METADATA_PATH = (
    PROJECT_ROOT / "quantization" / "quantization_metadata.json"
)

SCALES_PATH = (
    PROJECT_ROOT / "quantization" / "activation_scales.json"
)


# --------------------------------------------------
# 1. Load model with quantized weights
# --------------------------------------------------

def load_quantized_weight_model():

    model = CIFAR10_CNN().to(DEVICE)

    weights = torch.load(
        WEIGHTS_PATH,
        map_location="cpu",
        weights_only=True
    )

    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)

    state_dict = {}

    for name, tensor in weights.items():

        scale = metadata[name]["scale"]

        state_dict[name] = tensor.float() * scale

    model.load_state_dict(state_dict)
    model.eval()

    return model


# --------------------------------------------------
# 2. Calibrate activation ranges
# --------------------------------------------------

def calibrate_activations(model, calibration_loader):

    max_values = {
        "pool1": 0.0,
        "pool2": 0.0,
        "fc1_relu": 0.0
    }

    model.eval()

    with torch.no_grad():

        for images, _ in calibration_loader:

            images = images.to(DEVICE)

            # Conv1 -> ReLU -> Pool1
            x = model.pool(
                torch.relu(model.conv1(images))
            )

            max_values["pool1"] = max(
                max_values["pool1"],
                x.abs().max().item()
            )

            # Conv2 -> ReLU -> Pool2
            x = model.pool(
                torch.relu(model.conv2(x))
            )

            max_values["pool2"] = max(
                max_values["pool2"],
                x.abs().max().item()
            )

            # FC1 -> ReLU
            x = torch.flatten(x, start_dim=1)

            x = torch.relu(model.fc1(x))

            max_values["fc1_relu"] = max(
                max_values["fc1_relu"],
                x.abs().max().item()
            )

    # Symmetric signed INT8 quantization
    scales = {}

    for name, max_value in max_values.items():

        scales[name] = max(max_value / 127.0, 1e-8)

    return scales


# --------------------------------------------------
# 3. Save activation scales
# --------------------------------------------------

def save_activation_scales(scales):

    with open(SCALES_PATH, "w") as file:

        json.dump(
            {
                "activation_dtype": "int8",
                "quantization": "symmetric",
                "zero_point": 0,
                "scales": scales
            },
            file,
            indent=4
        )

    print("\nActivation scales saved.")


# --------------------------------------------------
# 4. Main
# --------------------------------------------------

def main():

    print("Using device:", DEVICE)

    print("\nLoading dataset...")

    _, calibration_loader, _ = get_dataloaders()

    print("\nLoading quantized-weight model...")

    model = load_quantized_weight_model()

    print("\nCalibrating activation ranges...")

    scales = calibrate_activations(
        model,
        calibration_loader
    )

    print("\n========== ACTIVATION QUANTIZATION ==========")

    for name, scale in scales.items():

        print(f"{name:15s}: {scale:.8f}")

    save_activation_scales(scales)

    print("\nActivation calibration completed!")


if __name__ == "__main__":
    main()
