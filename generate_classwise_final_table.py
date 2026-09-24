from ultralytics import YOLO
from pathlib import Path
import pandas as pd


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


class_names = [
    "rider",
    "no_helmet",
    "helmet"
]


rows = []


for model_name, model_path in MODELS.items():

    print("\n================")
    print(model_name)
    print("================")


    model = YOLO(model_path)


    result = model.val(
        data=DATA,
        split="val",
        workers=0,
        verbose=False
    )


    # per-class arrays

    precision = result.box.p
    recall = result.box.r
    map50 = result.box.ap50
    map5095 = result.box.ap


    for idx, cls in enumerate(class_names):

        rows.append(
            [
                model_name,
                cls,
                round(float(precision[idx]),4),
                round(float(recall[idx]),4),
                round(float(map50[idx]),4),
                round(float(map5095[idx]),4)
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


save_path = (
    OUTPUT /
    "table_classwise_comparison.csv"
)


df.to_csv(
    save_path,
    index=False
)


print("\nSaved:")
print(save_path)

print("\n")
print(df)