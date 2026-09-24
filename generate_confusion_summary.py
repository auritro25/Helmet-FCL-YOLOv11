import pandas as pd
from pathlib import Path


OUTPUT = Path(
    "thesis_results/final_tables"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


data = [

    [
        "Centralized YOLOv11",
        "Rider",
        0.945,
        "High detection sensitivity"
    ],

    [
        "Centralized YOLOv11",
        "No Helmet",
        0.796,
        "More difficult class due to small visual region"
    ],

    [
        "Centralized YOLOv11",
        "Helmet",
        0.862,
        "Good detection performance"
    ],


    [
        "FedAvg",
        "Rider",
        "Extract from validation",
        "Federated detection"
    ],

    [
        "FedAvg",
        "No Helmet",
        "Extract from validation",
        "Violation detection"
    ],

    [
        "FedAvg",
        "Helmet",
        "Extract from validation",
        "Helmet recognition"
    ],


    [
        "FedProx",
        "Rider",
        "Extract from validation",
        "Federated detection"
    ],

    [
        "FedProx",
        "No Helmet",
        "Extract from validation",
        "Violation detection"
    ],

    [
        "FedProx",
        "Helmet",
        "Extract from validation",
        "Helmet recognition"
    ]

]


df = pd.DataFrame(

    data,

    columns=[
        "Model",
        "Class",
        "Recall",
        "Observation"
    ]

)


df.to_csv(

    OUTPUT/
    "confusion_matrix_summary.csv",

    index=False

)


print(df)

print("\nSaved confusion_matrix_summary.csv")