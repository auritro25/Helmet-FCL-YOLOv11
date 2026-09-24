import pandas as pd
from pathlib import Path


output = Path(
    "thesis_results/tables"
)

output.mkdir(
    parents=True,
    exist_ok=True
)


data = {

    "Centralized YOLOv11": {
        "FL": "No",
        "FedProx": "No",
        "Precision": 0.858,
        "Recall": 0.868,
        "mAP50": 0.901,
        "mAP50-95": 0.630
    },

    "FedAvg": {
        "FL": "Yes",
        "FedProx": "No",
        "Precision": 0.868,
        "Recall": 0.851,
        "mAP50": 0.897,
        "mAP50-95": 0.625
    },

    "FedProx": {
        "FL": "Yes",
        "FedProx": "Yes",
        "Precision": 0.852,
        "Recall": 0.859,
        "mAP50": 0.887,
        "mAP50-95": 0.620
    }

}


df = pd.DataFrame(data).T


df.index.name = "Model"


df.to_csv(
    output / "ablation_results.csv"
)


# Calculate differences

comparison = pd.DataFrame({

    "Comparison": [
        "Centralized → FedAvg",
        "FedAvg → FedProx"
    ],

    "Δ Precision": [
        0.868 - 0.858,
        0.852 - 0.868
    ],

    "Δ Recall": [
        0.851 - 0.868,
        0.859 - 0.851
    ],

    "Δ mAP50": [
        0.897 - 0.901,
        0.887 - 0.897
    ],

    "Δ mAP50-95": [
        0.625 - 0.630,
        0.620 - 0.625
    ]

})


comparison.to_csv(
    output / "ablation_difference_analysis.csv",
    index=False
)


print("\nAblation Results")
print("================")
print(df)

print("\nDifference Analysis")
print("===================")
print(comparison)


print("\nSaved:")
print("ablation_results.csv")
print("ablation_difference_analysis.csv")