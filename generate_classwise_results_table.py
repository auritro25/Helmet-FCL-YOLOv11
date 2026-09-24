from ultralytics import YOLO
import pandas as pd
from pathlib import Path


MODELS = {

    "Centralized YOLOv11":
    r"runs\detect\runs\baseline\yolo11s_baseline-6\weights\best.pt",

    "FedAvg":
    r"federated\server\converted\fedavg_round5.pt",

    "FedProx":
    r"federated\server\converted\fedprox_round5.pt"

}


DATA = "datasets/merged/data.yaml"


OUTPUT = Path(
    "thesis_results/final_tables"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


classes = [
    "rider",
    "no_helmet",
    "helmet"
]


rows = []


for name, path in MODELS.items():

    print("\n================")
    print(name)
    print("================")


    model = YOLO(path)


    results = model.val(
        data=DATA,
        split="val",
        workers=0,
        verbose=False
    )


    precision = results.box.p
    recall = results.box.r
    map50 = results.box.ap50
    map5095 = results.box.ap


    for i, cls in enumerate(classes):

        rows.append(
            [
                name,
                cls,
                float(precision[i]),
                float(recall[i]),
                float(map50[i]),
                float(map5095[i])
            ]
        )


df = pd.DataFrame(

    rows,

    columns=[
        "Model",
        "Class",
        "Precision",
        "Recall",
        "mAP@0.5",
        "mAP@0.5:0.95"
    ]

)


df.to_csv(
    OUTPUT/"table_classwise_comparison.csv",
    index=False
)


print("\nSaved:")
print(
    OUTPUT/"table_classwise_comparison.csv"
)

print(df)