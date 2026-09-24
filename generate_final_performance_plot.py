import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


OUTPUT = Path(
    "thesis_results/graphs/final_comparison"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


data = {

    "Model":[
        "Centralized YOLOv11",
        "FedAvg",
        "FedProx"
    ],

    "Precision":[
        0.858,
        0.868,
        0.852
    ],

    "Recall":[
        0.868,
        0.851,
        0.859
    ],

    "mAP50":[
        0.901,
        0.897,
        0.887
    ],

    "mAP50-95":[
        0.630,
        0.625,
        0.620
    ]

}


df = pd.DataFrame(data)


metrics = [
    "Precision",
    "Recall",
    "mAP50",
    "mAP50-95"
]


for metric in metrics:

    plt.figure(
        figsize=(8,5)
    )


    plt.bar(
        df["Model"],
        df[metric]
    )


    plt.ylabel(metric)

    plt.xlabel(
        "Model"
    )


    plt.title(
        f"{metric} Comparison Across Training Strategies"
    )


    plt.xticks(
        rotation=20
    )


    plt.ylim(
        0,
        1
    )


    plt.tight_layout()


    plt.savefig(
        OUTPUT /
        f"{metric}_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


print("Final comparison graphs generated")