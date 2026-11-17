
import json
import torch

from pathlib import Path

from software.config import BEST_MODEL_PATH
from software.model import CIFAR10_CNN


ROOT = Path(__file__).resolve().parent

EXPORT_DIR = ROOT / "exported"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

INT32_MIN = -(2**31)
INT32_MAX = (2**31) - 1


# --------------------------------------------------
# 1. Load all required parameters
# --------------------------------------------------

def load_parameters():

    weights = torch.load(
        ROOT / "quantized_weights.pth",
        map_location="cpu",
        weights_only=True
    )

    with open(ROOT / "quantization_metadata.json") as file:
        weight_metadata = json.load(file)

    with open(ROOT / "input_scale.json") as file:
        input_metadata = json.load(file)

    with open(ROOT / "activation_scales.json") as file:
        activation_metadata = json.load(file)

    with open(ROOT / "fixed_point_parameters.json") as file:
        fixed_point_metadata = json.load(file)

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model = CIFAR10_CNN()
    model.load_state_dict(checkpoint["model_state_dict"])

    biases = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if name.endswith(".bias")
    }

    return (
        weights,
        biases,
        weight_metadata,
        input_metadata,
        activation_metadata,
        fixed_point_metadata
    )


# --------------------------------------------------
# 2. Convert signed integer to hexadecimal
# --------------------------------------------------

def to_hex(value, bits):

    value = int(value)

    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1

    if not minimum <= value <= maximum:
        raise OverflowError(
            f"{value} does not fit in signed INT{bits}"
        )

    value &= (1 << bits) - 1

    digits = bits // 4

    return f"{value:0{digits}X}"


# --------------------------------------------------
# 3. Export tensor into memory initialization file
# --------------------------------------------------

def export_tensor(tensor, filepath, bits):

    tensor = tensor.detach().cpu().contiguous().reshape(-1)

    with open(filepath, "w") as file:

        for value in tensor.tolist():

            file.write(
                to_hex(value, bits) + "\n"
            )

    print(
        f"Exported {filepath.name}: "
        f"{tensor.numel()} values"
    )


# --------------------------------------------------
# 4. Quantize bias to INT32
# --------------------------------------------------

def quantize_bias(bias, bias_scale):

    if bias_scale <= 0:
        raise ValueError("Bias scale must be positive.")

    quantized = torch.round(
        bias.to(torch.float64) / bias_scale
    ).to(torch.int64)

    if torch.any(quantized < INT32_MIN) or torch.any(
        quantized > INT32_MAX
    ):
        raise OverflowError("INT32 bias overflow detected.")

    return quantized


# --------------------------------------------------
# 5. Main export process
# --------------------------------------------------

def main():

    print("Loading model parameters...")

    (
        weights,
        biases,
        weight_metadata,
        input_metadata,
        activation_metadata,
        fixed_point_metadata
    ) = load_parameters()

    input_scale = input_metadata["scale"]

    activation_scales = activation_metadata["scales"]

    # Input scale for each layer
    layer_input_scales = {
        "conv1": input_scale,
        "conv2": activation_scales["pool1"],
        "fc1": activation_scales["pool2"],
        "fc2": activation_scales["fc1_relu"]
    }

    layer_metadata = {}

    print("\nExporting layer parameters...\n")

    for layer in ["conv1", "conv2", "fc1", "fc2"]:

        weight_key = f"{layer}.weight"
        bias_key = f"{layer}.bias"

        quantized_weight = weights[weight_key]

        weight_scale = weight_metadata[weight_key]["scale"]

        bias_scale = (
            layer_input_scales[layer] * weight_scale
        )

        quantized_bias = quantize_bias(
            biases[bias_key],
            bias_scale
        )

        # Export weights
        export_tensor(
            quantized_weight,
            EXPORT_DIR / f"{layer}_weights.mem",
            8
        )

        # Export biases
        export_tensor(
            quantized_bias,
            EXPORT_DIR / f"{layer}_bias.mem",
            32
        )

        layer_metadata[layer] = {
            "weight_shape": list(quantized_weight.shape),
            "bias_shape": list(quantized_bias.shape),
            "weight_count": quantized_weight.numel(),
            "bias_count": quantized_bias.numel(),
            "weight_dtype": "INT8",
            "bias_dtype": "INT32",
            "weight_scale": weight_scale,
            "input_scale": layer_input_scales[layer],
            "bias_scale": bias_scale,
            "weight_layout": (
                "OIHW" if layer.startswith("conv") else "OI"
            ),
            "flattening_order": "C_ORDER"
        }

        if layer in fixed_point_metadata:
            layer_metadata[layer]["requantization"] = (
                fixed_point_metadata[layer]
            )

    # --------------------------------------------------
    # Save complete hardware metadata
    # --------------------------------------------------

    hardware_metadata = {
        "model": "CIFAR10_CNN",
        "dataset": "CIFAR-10",
        "weight_precision": "INT8",
        "activation_precision": "INT8",
        "bias_precision": "INT32",
        "input_scale": input_scale,
        "activation_scales": activation_scales,
        "layers": layer_metadata
    }

    metadata_path = EXPORT_DIR / "hardware_metadata.json"

    with open(metadata_path, "w") as file:
        json.dump(
            hardware_metadata,
            file,
            indent=4
        )

    print("\n======================================")
    print("PARAMETER EXPORT COMPLETED")
    print("======================================")

    print(f"Export directory: {EXPORT_DIR}")
    print(f"Metadata file   : {metadata_path}")


if __name__ == "__main__":
    main()
