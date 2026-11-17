
import torch
import torch.nn as nn
import json
import pandas as pd

from pathlib import Path
from sklearn.metrics import classification_report

from software.config import (
    DEVICE,
    BEST_MODEL_PATH,
    CLASS_NAMES
)

from software.dataset import get_dataloaders
from software.model import CIFAR10_CNN


PROJECT_ROOT = Path(__file__).resolve().parent.parent

WEIGHTS_PATH = PROJECT_ROOT / "quantization" / "quantized_weights.pth"
METADATA_PATH = PROJECT_ROOT / "quantization" / "quantization_metadata.json"
SCALES_PATH = PROJECT_ROOT / "quantization" / "activation_scales.json"

RESULTS_DIR = PROJECT_ROOT / "results" / "accuracy"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. Load original FP32 model
# --------------------------------------------------

def load_fp32_model():

    model = CIFAR10_CNN().to(DEVICE)

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


# --------------------------------------------------
# 2. Load model with quantized weights
# --------------------------------------------------

def load_weight_quantized_model():

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
# 3. Load activation scales
# --------------------------------------------------

def load_activation_scales():

    with open(SCALES_PATH, "r") as file:
        metadata = json.load(file)

    return metadata["scales"]


# --------------------------------------------------
# 4. Activation quantization simulation
# --------------------------------------------------

def quantize_dequantize(x, scale):

    quantized = torch.round(x / scale)

    quantized = torch.clamp(
        quantized,
        -128,
        127
    )

    return quantized * scale


# --------------------------------------------------
# 5. Forward pass with activation quantization
# --------------------------------------------------

def quantized_activation_forward(model, images, scales):

    # Conv1 -> ReLU -> Pool1
    x = model.conv1(images)
    x = torch.relu(x)
    x = model.pool(x)

    x = quantize_dequantize(
        x,
        scales["pool1"]
    )

    # Conv2 -> ReLU -> Pool2
    x = model.conv2(x)
    x = torch.relu(x)
    x = model.pool(x)

    x = quantize_dequantize(
        x,
        scales["pool2"]
    )

    # Flatten
    x = torch.flatten(x, start_dim=1)

    # FC1 -> ReLU
    x = model.fc1(x)
    x = torch.relu(x)

    x = quantize_dequantize(
        x,
        scales["fc1_relu"]
    )

    # FC2
    x = model.fc2(x)

    return x


# --------------------------------------------------
# 6. Evaluate model
# --------------------------------------------------

def evaluate_model(model, test_loader, scales=None):

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    model.eval()

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if scales is None:

                outputs = model(images)

            else:

                outputs = quantized_activation_forward(
                    model,
                    images,
                    scales
                )

            loss = criterion(outputs, labels)

            predictions = torch.argmax(outputs, dim=1)

            total_loss += loss.item() * images.size(0)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    accuracy = 100.0 * correct / total
    average_loss = total_loss / total

    return average_loss, accuracy, all_labels, all_predictions


# --------------------------------------------------
# 7. Save comparison
# --------------------------------------------------

def save_results(results):

    with open(
        RESULTS_DIR / "activation_quantization_results.json",
        "w"
    ) as file:

        json.dump(results, file, indent=4)

    dataframe = pd.DataFrame([
        {
            "Model": name,
            "Test Accuracy (%)": values["accuracy"],
            "Test Loss": values["loss"]
        }
        for name, values in results.items()
    ])

    dataframe.to_csv(
        RESULTS_DIR / "activation_quantization_comparison.csv",
        index=False
    )

    print("\nResults saved successfully.")


# --------------------------------------------------
# 8. Main
# --------------------------------------------------

def main():

    print("Using device:", DEVICE)

    print("\nLoading test dataset...")

    _, _, test_loader = get_dataloaders()

    print("\nLoading models...")

    fp32_model = load_fp32_model()
    quantized_model = load_weight_quantized_model()

    scales = load_activation_scales()

    results = {}

    # FP32
    print("\nEvaluating FP32 model...")

    loss, accuracy, _, _ = evaluate_model(
        fp32_model,
        test_loader
    )

    results["FP32"] = {
        "loss": loss,
        "accuracy": accuracy
    }

    # Weight quantization
    print("\nEvaluating weight-quantized model...")

    loss, accuracy, _, _ = evaluate_model(
        quantized_model,
        test_loader
    )

    results["Weight INT8"] = {
        "loss": loss,
        "accuracy": accuracy
    }

    # Weight + activation quantization
    print("\nEvaluating weight + activation quantization...")

    loss, accuracy, labels, predictions = evaluate_model(
        quantized_model,
        test_loader,
        scales
    )

    results["Weight + Activation INT8"] = {
        "loss": loss,
        "accuracy": accuracy
    }

    # Display results
    print("\n========== QUANTIZATION COMPARISON ==========")

    for name, values in results.items():

        print(f"\n{name}")
        print(f"Test Loss     : {values['loss']:.4f}")
        print(f"Test Accuracy : {values['accuracy']:.2f}%")

    print("\n========== ACTIVATION QUANTIZED CLASSIFICATION REPORT ==========")

    print(classification_report(
        labels,
        predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    save_results(results)


if __name__ == "__main__":
    main()
