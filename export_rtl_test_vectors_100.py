"""
Export 100 CIFAR-10 test images and their IntegerCNN reference logits.

Run from the quantization directory:
    PYTHONPATH=.. python export_rtl_test_vectors_100.py

Outputs:
    input_images_100.mem       100 * 3072 signed INT8 values, image-major CHW
    expected_logits_100.mem    100 * 10 signed INT32 logits, image-major
    test_images_100_info.txt   labels and Python predictions
"""

import torch

from integer_inference import IntegerCNN
from software.dataset import get_dataloaders

NUM_IMAGES = 100
IMAGE_VALUES = 3 * 32 * 32
NUM_CLASSES = 10


def to_hex(value: int, bits: int) -> str:
    """Encode signed integer as a fixed-width two's-complement hex word."""
    return f"{(int(value) & ((1 << bits) - 1)):0{bits // 4}X}"


def main():
    print("Loading CIFAR-10 test dataset...")
    _, _, test_loader = get_dataloaders()

    print(f"Collecting the first {NUM_IMAGES} test images...")
    image_batches = []
    label_batches = []
    collected = 0

    for images, labels in test_loader:
        take = min(images.shape[0], NUM_IMAGES - collected)
        image_batches.append(images[:take].cpu())
        label_batches.append(labels[:take].cpu())
        collected += take
        if collected == NUM_IMAGES:
            break

    if collected != NUM_IMAGES:
        raise RuntimeError(f"Only collected {collected} images")

    images = torch.cat(image_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)

    print("Initializing IntegerCNN...")
    model = IntegerCNN()

    # Export the exact quantized tensor used by IntegerCNN.forward().
    quantized_inputs = model.quantize(images, model.input_scale)
    if tuple(quantized_inputs.shape) != (NUM_IMAGES, 3, 32, 32):
        raise RuntimeError(
            f"Unexpected quantized input shape: {tuple(quantized_inputs.shape)}"
        )

    # Run the same reference implementation used for the single-image test.
    print("Running integer inference for 100 images...")
    with torch.no_grad():
        logits = model.forward(images)

    logits = logits.to(torch.int64).cpu()
    if tuple(logits.shape) != (NUM_IMAGES, NUM_CLASSES):
        raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

    inputs_list = quantized_inputs.to(torch.int32).contiguous().view(-1).tolist()
    logits_list = logits.contiguous().view(-1).tolist()

    for index, value in enumerate(inputs_list):
        if not -127 <= int(value) <= 127:
            raise OverflowError(f"Input value {index} outside signed INT8")

    for index, value in enumerate(logits_list):
        if not -(2**31) <= int(value) <= (2**31 - 1):
            raise OverflowError(f"Logit value {index} outside signed INT32")

    with open("input_images_100.mem", "w", encoding="ascii") as f:
        for value in inputs_list:
            f.write(to_hex(value, 8) + "\n")

    with open("expected_logits_100.mem", "w", encoding="ascii") as f:
        for value in logits_list:
            f.write(to_hex(value, 32) + "\n")

    predictions = torch.argmax(logits, dim=1)
    correct = int((predictions == labels).sum().item())

    with open("test_images_100_info.txt", "w", encoding="utf-8") as f:
        f.write(f"Images exported: {NUM_IMAGES}\n")
        f.write(f"Input values per image: {IMAGE_VALUES}\n")
        f.write(f"Logits per image: {NUM_CLASSES}\n")
        f.write(f"IntegerCNN accuracy on this subset: {100.0 * correct / NUM_IMAGES:.2f}%\n\n")
        for image_id in range(NUM_IMAGES):
            f.write(
                f"Image {image_id:03d}: "
                f"label={int(labels[image_id])}, "
                f"predicted={int(predictions[image_id])}, "
                f"logits={[int(v) for v in logits[image_id].tolist()]}\n"
            )

    print("\n100-image RTL vectors exported successfully.")
    print(f"Input values : {len(inputs_list)} (INT8)")
    print(f"Expected logits: {len(logits_list)} (INT32)")
    print(f"Correct predictions in subset: {correct}/{NUM_IMAGES}")
    print("\nCreated:")
    print("  input_images_100.mem")
    print("  expected_logits_100.mem")
    print("  test_images_100_info.txt")


if __name__ == "__main__":
    main()
