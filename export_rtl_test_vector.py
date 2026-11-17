"""
Export one CIFAR-10 test image and its integer-CNN logits for RTL simulation.

Place this file beside integer_inference.py in the quantization folder.
Run from the project environment:
    python export_rtl_test_vector.py

Outputs:
    input_image.mem
    expected_logits.mem
    test_image_info.txt
"""

import torch

from integer_inference import IntegerCNN
from software.dataset import get_dataloaders


def to_hex(value: int, bits: int) -> str:
    """Encode a signed integer as a two's-complement hex word."""
    return f"{(int(value) & ((1 << bits) - 1)):0{bits // 4}X}"


def main():
    print("Loading CIFAR-10 test dataset...")
    _, _, test_loader = get_dataloaders()

    print("Initializing the same IntegerCNN used by integer_inference.py...")
    model = IntegerCNN()

    # Use the first test image, without changing the dataset preprocessing.
    images, labels = next(iter(test_loader))
    image = images[0:1].cpu()
    label = int(labels[0].item())

    # Export the exact quantized input tensor used by IntegerCNN.forward().
    quantized_input = model.quantize(image, model.input_scale)
    input_values = quantized_input[0].to(torch.int32).contiguous().view(-1)

    if input_values.numel() != 3 * 32 * 32:
        raise RuntimeError(
            f"Expected 3072 input values, got {input_values.numel()}"
        )

    # Run the reference integer inference on the same image.
    with torch.no_grad():
        logits = model.forward(image)

    logits = logits[0].to(torch.int64).cpu().tolist()

    if len(logits) != 10:
        raise RuntimeError(f"Expected 10 logits, got {len(logits)}")

    # Check that all values fit the RTL interface widths.
    for index, value in enumerate(input_values.tolist()):
        if not -127 <= int(value) <= 127:
            raise OverflowError(f"Input value {index} outside signed INT8")

    for index, value in enumerate(logits):
        if not -(2**31) <= int(value) <= (2**31 - 1):
            raise OverflowError(f"Logit {index} outside signed INT32")

    predicted = max(range(10), key=lambda i: logits[i])

    # Tensor is [C,H,W]; flattening preserves the top-level CHW stream order.
    with open("input_image.mem", "w", encoding="ascii") as f:
        for value in input_values.tolist():
            f.write(to_hex(int(value), 8) + "\n")

    with open("expected_logits.mem", "w", encoding="ascii") as f:
        for value in logits:
            f.write(to_hex(value, 32) + "\n")

    with open("test_image_info.txt", "w", encoding="utf-8") as f:
        f.write(f"Dataset label: {label}\n")
        f.write(f"Python predicted class: {predicted}\n")
        f.write(f"Input scale: {model.input_scale}\n")
        f.write("Class logits (signed INT32):\n")
        for class_id, value in enumerate(logits):
            f.write(f"  class {class_id}: {value}\n")

    print("\nRTL test vector exported successfully.")
    print(f"Input values : {input_values.numel()} (INT8, CHW)")
    print(f"Expected logits: {len(logits)} (INT32)")
    print(f"Dataset label : {label}")
    print(f"Predicted class: {predicted}")
    print("Logits:")
    for class_id, value in enumerate(logits):
        print(f"  class {class_id}: {value}")

    print("\nCreated:")
    print("  input_image.mem")
    print("  expected_logits.mem")
    print("  test_image_info.txt")


if __name__ == "__main__":
    main()
