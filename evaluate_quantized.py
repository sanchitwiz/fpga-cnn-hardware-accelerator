
import torch
import torch.nn as nn
import json
import pandas as pd

from pathlib import Path
from sklearn.metrics import classification_report

from software.config import (
    DEVICE,
    BEST_MODEL_PATH,
    ACCURACY_RESULTS_DIR,
    CLASS_NAMES
)

from software.dataset import get_dataloaders
from software.model import CIFAR10_CNN


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUANTIZED_WEIGHTS_PATH = (
    PROJECT_ROOT / "quantization" / "quantized_weights.pth"
)

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

    print("FP32 model loaded successfully.")

    return model


# --------------------------------------------------
# 2. Load quantized weights
# --------------------------------------------------

def load_quantized_model():

    model = CIFAR10_CNN().to(DEVICE)

    quantized_weights = torch.load(
        QUANTIZED_WEIGHTS_PATH,
        map_location="cpu",
        weights_only=True
    )

    dequantized_state_dict = {}

    for name, quantized_tensor in quantized_weights.items():

        metadata_path = (
            PROJECT_ROOT
            / "quantization"
            / "quantization_metadata.json"
        )

        with open(metadata_path, "r") as file:
            metadata = json.load(file)

        scale = metadata[name]["scale"]

        dequantized_tensor = (
            quantized_tensor.float() * scale
        )

        dequantized_state_dict[name] = dequantized_tensor

    model.load_state_dict(dequantized_state_dict)

    model.eval()

    print("Quantized-weight model loaded successfully.")

    return model


# --------------------------------------------------
# 3. Evaluate model
# --------------------------------------------------

def evaluate_model(model, test_loader):

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

            outputs = model(images)

            loss = criterion(outputs, labels)

            predictions = torch.argmax(outputs, dim=1)

            total_loss += loss.item() * images.size(0)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return (
        average_loss,
        accuracy,
        all_labels,
        all_predictions
    )


# --------------------------------------------------
# 4. Save results
# --------------------------------------------------

def save_results(fp32_accuracy, quantized_accuracy):

    accuracy_drop = fp32_accuracy - quantized_accuracy

    results = {
        "fp32_accuracy": fp32_accuracy,
        "quantized_weight_accuracy": quantized_accuracy,
        "accuracy_drop_percentage_points": accuracy_drop
    }

    output_path = RESULTS_DIR / "quantization_comparison.json"

    with open(output_path, "w") as file:
        json.dump(results, file, indent=4)

    comparison = pd.DataFrame({
        "Model": [
            "FP32",
            "INT8 Quantized Weights"
        ],
        "Test Accuracy (%)": [
            fp32_accuracy,
            quantized_accuracy
        ]
    })

    comparison.to_csv(
        RESULTS_DIR / "quantization_comparison.csv",
        index=False
    )

    print("\nComparison results saved successfully.")


# --------------------------------------------------
# 5. Main
# --------------------------------------------------

def main():

    print("Using device:", DEVICE)

    print("\nLoading test dataset...")

    _, _, test_loader = get_dataloaders()

    # FP32 evaluation
    print("\nEvaluating FP32 model...")

    fp32_model = load_fp32_model()

    fp32_loss, fp32_accuracy, _, _ = evaluate_model(
        fp32_model,
        test_loader
    )

    # Quantized weight evaluation
    print("\nEvaluating quantized-weight model...")

    quantized_model = load_quantized_model()

    quantized_loss, quantized_accuracy, labels, predictions = (
        evaluate_model(
            quantized_model,
            test_loader
        )
    )

    # Results
    print("\n========== QUANTIZATION RESULTS ==========")

    print(
        f"FP32 Test Loss          : {fp32_loss:.4f}"
    )

    print(
        f"FP32 Test Accuracy      : {fp32_accuracy:.2f}%"
    )

    print(
        f"Quantized Test Loss     : {quantized_loss:.4f}"
    )

    print(
        f"Quantized Test Accuracy : {quantized_accuracy:.2f}%"
    )

    print(
        f"Accuracy Drop           : "
        f"{fp32_accuracy - quantized_accuracy:.2f} percentage points"
    )

    print("\n========== QUANTIZED CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            labels,
            predictions,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    save_results(
        fp32_accuracy,
        quantized_accuracy
    )


if __name__ == "__main__":
    main()
