
import torch
import torch.nn as nn
import json

from software.config import (
    DEVICE,
    BEST_MODEL_PATH,
    CLASS_NAMES
)

from software.model import CIFAR10_CNN


# --------------------------------------------------
# 1. Load trained FP32 model
# --------------------------------------------------

def load_model():

    model = CIFAR10_CNN().to(DEVICE)

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("FP32 model loaded successfully.")

    return model


# --------------------------------------------------
# 2. Symmetric INT8 weight quantization
# --------------------------------------------------

def quantize_tensor(tensor):

    qmax = 127

    max_value = torch.max(torch.abs(tensor))

    scale = max_value / qmax

    if scale == 0:
        scale = torch.tensor(1.0, device=tensor.device)

    quantized = torch.round(tensor / scale)

    quantized = torch.clamp(
        quantized,
        -128,
        127
    ).to(torch.int8)

    return quantized, scale.item()


# --------------------------------------------------
# 3. Quantize model weights
# --------------------------------------------------


def quantize_model(model):

    quantized_weights = {}
    quantization_metadata = {}

    for name, parameter in model.named_parameters():

        # Quantize weights only
        if name.endswith(".weight"):

            quantized_tensor, scale = quantize_tensor(
                parameter.detach()
            )

            quantized_weights[name] = quantized_tensor.cpu()

            quantization_metadata[name] = {
                "scale": scale,
                "zero_point": 0,
                "dtype": "int8",
                "shape": list(parameter.shape)
            }

            print(
                f"{name:20s} | "
                f"Scale: {scale:.8f}"
            )

    return quantized_weights, quantization_metadata

# --------------------------------------------------
# 4. Save quantized parameters
# --------------------------------------------------

def save_quantized_model(weights, metadata):

    output_dir = "quantization"

    torch.save(
        weights,
        f"{output_dir}/quantized_weights.pth"
    )

    with open(
        f"{output_dir}/quantization_metadata.json",
        "w"
    ) as file:

        json.dump(metadata, file, indent=4)

    print("\nQuantized weights saved successfully.")


# --------------------------------------------------
# 5. Main
# --------------------------------------------------

def main():

    print("Using device:", DEVICE)

    model = load_model()

    print("\nQuantizing model weights...\n")

    weights, metadata = quantize_model(model)

    save_quantized_model(weights, metadata)

    print("\nQuantization completed!")


if __name__ == "__main__":
    main()
