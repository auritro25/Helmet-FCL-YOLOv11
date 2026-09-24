import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


INPUT = Path(
    "thesis_results/tables/ablation_results.csv"
)


OUTPUT = Path(
    "thesis_results/graphs/ablation"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(INPUT)


df = df.set_index("Model")


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
        df.index,
        df[metric]
    )


    plt.ylabel(metric)

    plt.xlabel(
        "Training Approach"
    )


    plt.title(
        f"Ablation Study: {metric} Comparison"
    )


    plt.xticks(
        rotation=20
    )


    plt.tight_layout()


    save = OUTPUT / (
        metric.lower()
        +
        "_ablation.png"
    )


    plt.savefig(
        save,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "Saved:",
        save
    )


print("\nAblation graphs complete")