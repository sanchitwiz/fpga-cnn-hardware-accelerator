
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from tqdm import tqdm

from software.config import BEST_MODEL_PATH
from software.dataset import get_dataloaders
from software.model import CIFAR10_CNN

from quantization.fixed_point import (
    fixed_point_requantize,
    quantize_multiplier
)


ROOT = Path(__file__).resolve().parent

QMIN = -127
QMAX = 127

INT32_MIN = -(2**31)
INT32_MAX = (2**31) - 1


class IntegerCNN:

    def __init__(self):

        # ------------------------------------------
        # Load quantization scales
        # ------------------------------------------

        with open(ROOT / "input_scale.json") as file:
            self.input_scale = json.load(file)["scale"]

        with open(ROOT / "activation_scales.json") as file:
            self.scales = json.load(file)["scales"]

        with open(ROOT / "quantization_metadata.json") as file:
            self.metadata = json.load(file)

        # ------------------------------------------
        # Load INT8 weights
        # ------------------------------------------

        self.weights = torch.load(
            ROOT / "quantized_weights.pth",
            map_location="cpu",
            weights_only=True
        )

        # ------------------------------------------
        # Load original FP32 biases
        # ------------------------------------------

        checkpoint = torch.load(
            BEST_MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )

        fp32_model = CIFAR10_CNN()

        fp32_model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.biases = {
            name: parameter.detach().cpu()
            for name, parameter in fp32_model.named_parameters()
            if name.endswith(".bias")
        }

        # Fixed-point configuration
        self.fixed_point_params = {}

        # Overflow statistics
        self.layer_statistics = {}

    # ----------------------------------------------
    # Input quantization
    # ----------------------------------------------

    @staticmethod
    def quantize(tensor, scale):

        return torch.clamp(
            torch.round(tensor / scale),
            QMIN,
            QMAX
        ).to(torch.int32)

    # ----------------------------------------------
    # INT32 range checking
    # ----------------------------------------------

    @staticmethod
    def check_int32(tensor, layer_name, value_name):

        if torch.any(tensor < INT32_MIN) or torch.any(
            tensor > INT32_MAX
        ):

            raise OverflowError(
                f"{value_name} INT32 overflow in {layer_name}"
            )

    # ----------------------------------------------
    # Fixed-point requantization
    # ----------------------------------------------

    @staticmethod
    def requantize(accumulator, multiplier, relu=False):

        result = fixed_point_requantize(
            accumulator,
            multiplier
        )

        if relu:
            result = torch.clamp(result, min=0)

        result = torch.clamp(
            result,
            QMIN,
            QMAX
        )

        return result.to(torch.int32)

    # ----------------------------------------------
    # Integer convolution / fully connected layer
    # ----------------------------------------------

    def integer_layer(
        self,
        x,
        name,
        input_scale,
        output_scale=None,
        relu=False
    ):

        # INT8 weights represented as INT32
        weight = self.weights[
            f"{name}.weight"
        ].to(torch.int32)

        weight_scale = self.metadata[
            f"{name}.weight"
        ]["scale"]

        # ------------------------------------------
        # Quantize bias into INT32 accumulator scale
        # ------------------------------------------

        bias_scale = input_scale * weight_scale

        bias = torch.round(
            self.biases[f"{name}.bias"].to(torch.float64)
            / bias_scale
        ).to(torch.int64)

        self.check_int32(
            bias,
            name,
            "Bias"
        )

        # ------------------------------------------
        # Convolution
        # ------------------------------------------

        if name.startswith("conv"):

            patches = F.unfold(
                x.to(torch.float64),
                kernel_size=3,
                stride=1,
                padding=0
            ).to(torch.int64)

            batch, _, locations = patches.shape

            weight_matrix = weight.reshape(
                weight.shape[0],
                -1
            ).to(torch.int64)

            accumulator = torch.matmul(
                weight_matrix,
                patches
            )

            accumulator += bias.view(
                1, -1, 1
            )

            side = int(locations ** 0.5)

            accumulator = accumulator.reshape(
                batch,
                weight.shape[0],
                side,
                side
            )

        # ------------------------------------------
        # Fully connected layer
        # ------------------------------------------

        else:

            accumulator = torch.matmul(
                x.to(torch.int64),
                weight.to(torch.int64).T
            )

            accumulator += bias.view(1, -1)

        # ------------------------------------------
        # Check accumulator bounds
        # ------------------------------------------

        self.check_int32(
            accumulator,
            name,
            "Accumulator"
        )

        self.layer_statistics[name] = {
            "accumulator_min": int(accumulator.min()),
            "accumulator_max": int(accumulator.max()),
            "bias_min": int(bias.min()),
            "bias_max": int(bias.max())
        }

        # ------------------------------------------
        # Final FC layer: retain accumulator values
        # ------------------------------------------

        if output_scale is None:

            return accumulator

        # ------------------------------------------
        # Calculate fixed-point multiplier
        # ------------------------------------------

        real_multiplier = bias_scale / output_scale

        integer_multiplier, shift = quantize_multiplier(
            real_multiplier
        )

        self.fixed_point_params[name] = {
            "input_scale": input_scale,
            "weight_scale": weight_scale,
            "bias_scale": bias_scale,
            "output_scale": output_scale,
            "real_multiplier": real_multiplier,
            "integer_multiplier": integer_multiplier,
            "right_shift": shift
        }

        # ------------------------------------------
        # Requantization
        # ------------------------------------------

        output = self.requantize(
            accumulator,
            real_multiplier,
            relu=relu
        )

        return output

    # ----------------------------------------------
    # Complete CNN forward pass
    # ----------------------------------------------

    def forward(self, images):

        # Input INT8
        x = self.quantize(
            images.cpu(),
            self.input_scale
        )

        # ------------------------------------------
        # Conv1 -> ReLU -> Pool1
        # ------------------------------------------

        x = self.integer_layer(
            x,
            "conv1",
            self.input_scale,
            self.scales["pool1"],
            relu=True
        )

        x = F.max_pool2d(
            x,
            kernel_size=2,
            stride=2
        )

        # ------------------------------------------
        # Conv2 -> ReLU -> Pool2
        # ------------------------------------------

        x = self.integer_layer(
            x,
            "conv2",
            self.scales["pool1"],
            self.scales["pool2"],
            relu=True
        )

        x = F.max_pool2d(
            x,
            kernel_size=2,
            stride=2
        )

        # ------------------------------------------
        # Flatten
        # ------------------------------------------

        x = torch.flatten(
            x,
            start_dim=1
        )

        # ------------------------------------------
        # FC1 -> ReLU
        # ------------------------------------------

        x = self.integer_layer(
            x,
            "fc1",
            self.scales["pool2"],
            self.scales["fc1_relu"],
            relu=True
        )

        # ------------------------------------------
        # FC2 -> INT32 logits
        # ------------------------------------------

        logits = self.integer_layer(
            x,
            "fc2",
            self.scales["fc1_relu"]
        )

        return logits


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def main():

    print("\nLoading CIFAR-10 test dataset...")

    _, _, test_loader = get_dataloaders()

    print("\nInitializing integer CNN...")

    model = IntegerCNN()

    correct = 0
    total = 0

    print("\nRunning integer inference...")

    with torch.no_grad():

        for images, labels in tqdm(test_loader):

            logits = model.forward(images)

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = 100.0 * correct / total

    print("\n========== INTEGER CNN RESULTS ==========")

    print(f"Test Images   : {total}")
    print(f"Correct       : {correct}")
    print(f"Incorrect     : {total - correct}")
    print(f"Test Accuracy : {accuracy:.2f}%")

    # ------------------------------------------
    # Save fixed-point parameters
    # ------------------------------------------

    with open(
        ROOT / "fixed_point_parameters.json",
        "w"
    ) as file:

        json.dump(
            model.fixed_point_params,
            file,
            indent=4
        )

    # ------------------------------------------
    # Save accumulator statistics
    # ------------------------------------------

    with open(
        ROOT / "accumulator_statistics.json",
        "w"
    ) as file:

        json.dump(
            model.layer_statistics,
            file,
            indent=4
        )

    # ------------------------------------------
    # Save evaluation results
    # ------------------------------------------

    output = {
        "test_images": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": accuracy,
        "arithmetic": "INT8 operands, INT64 intermediate accumulation with INT32 range checks",
        "requantization": "Q31 fixed-point multiplier and shift",
        "accumulator_validation": "INT32 range checked"
    }

    results_dir = (
        ROOT.parent / "results" / "accuracy"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        results_dir / "integer_inference_results.json",
        "w"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    print("\nGenerated files:")
    print("- quantization/fixed_point_parameters.json")
    print("- quantization/accumulator_statistics.json")
    print("- results/accuracy/integer_inference_results.json")

    print("\nInteger inference completed!")


if __name__ == "__main__":
    main()
