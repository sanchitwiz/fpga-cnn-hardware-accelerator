"""
Export Python intermediate activations for one CIFAR-10 image.
Run from quantization:
    PYTHONPATH=.. python export_intermediate_reference.py

Creates:
    input_image_intermediate.mem
    expected_conv1.mem
    expected_pool1.mem
    expected_conv2.mem
    expected_pool2.mem
    expected_fc1.mem
    expected_fc2.mem
    intermediate_reference_info.txt
"""

import torch
import torch.nn.functional as F

from integer_inference import IntegerCNN
from software.dataset import get_dataloaders


def to_hex(value: int, bits: int) -> str:
    return f"{(int(value) & ((1 << bits) - 1)):0{bits // 4}X}"


def write_mem(path, values, bits):
    values = values.reshape(-1).tolist()
    with open(path, "w", encoding="ascii") as f:
        for value in values:
            f.write(to_hex(int(value), bits) + "\n")
    return len(values)


def main():
    print("Loading CIFAR-10 test dataset...")
    _, _, test_loader = get_dataloaders()
    images, labels = next(iter(test_loader))

    image = images[0:1].cpu()
    label = int(labels[0].item())

    print("Initializing IntegerCNN...")
    model = IntegerCNN()

    # Keep every operation identical to IntegerCNN.forward().
    with torch.no_grad():
        x_input = model.quantize(image, model.input_scale)

        x_conv1 = model.integer_layer(
            x_input,
            "conv1",
            model.input_scale,
            model.scales["pool1"],
            relu=True
        )

        x_pool1 = F.max_pool2d(x_conv1, kernel_size=2, stride=2)

        x_conv2 = model.integer_layer(
            x_pool1,
            "conv2",
            model.scales["pool1"],
            model.scales["pool2"],
            relu=True
        )

        x_pool2 = F.max_pool2d(x_conv2, kernel_size=2, stride=2)

        x_flat = torch.flatten(x_pool2, start_dim=1)

        x_fc1 = model.integer_layer(
            x_flat,
            "fc1",
            model.scales["pool2"],
            model.scales["fc1_relu"],
            relu=True
        )

        x_fc2 = model.integer_layer(
            x_fc1,
            "fc2",
            model.scales["fc1_relu"]
        )

    # Sanity-check shapes before exporting.
    expected_shapes = {
        "input": (1, 3, 32, 32),
        "conv1": (1, 32, 30, 30),
        "pool1": (1, 32, 15, 15),
        "conv2": (1, 64, 13, 13),
        "pool2": (1, 64, 6, 6),
        "fc1": (1, 128),
        "fc2": (1, 10),
    }

    tensors = {
        "input": x_input,
        "conv1": x_conv1,
        "pool1": x_pool1,
        "conv2": x_conv2,
        "pool2": x_pool2,
        "fc1": x_fc1,
        "fc2": x_fc2,
    }

    for name, tensor in tensors.items():
        if tuple(tensor.shape) != expected_shapes[name]:
            raise RuntimeError(
                f"{name}: expected {expected_shapes[name]}, got {tuple(tensor.shape)}"
            )

    # Export input in CHW order. Intermediate feature maps are NCHW.
    write_mem("input_image_intermediate.mem", x_input[0], 8)
    write_mem("expected_conv1.mem", x_conv1[0], 8)
    write_mem("expected_pool1.mem", x_pool1[0], 8)
    write_mem("expected_conv2.mem", x_conv2[0], 8)
    write_mem("expected_pool2.mem", x_pool2[0], 8)
    write_mem("expected_fc1.mem", x_fc1[0], 8)
    write_mem("expected_fc2.mem", x_fc2[0], 32)

    prediction = int(torch.argmax(x_fc2, dim=1).item())

    with open("intermediate_reference_info.txt", "w", encoding="utf-8") as f:
        f.write(f"Dataset label: {label}\n")
        f.write(f"Python predicted class: {prediction}\n")
        f.write(f"Input scale: {model.input_scale}\n")
        f.write(f"Activation scales: {model.scales}\n")
        for name, tensor in tensors.items():
            t = tensor.to(torch.int64)
            f.write(
                f"{name}: shape={tuple(tensor.shape)}, "
                f"min={int(t.min())}, max={int(t.max())}\n"
            )

    print("\nIntermediate reference files exported.")
    for name, tensor in tensors.items():
        print(f"{name:>7}: shape={tuple(tensor.shape)}")
    print(f"Dataset label: {label}")
    print(f"Predicted class: {prediction}")
    print("Created input + six expected output memory files.")


if __name__ == "__main__":
    main()
