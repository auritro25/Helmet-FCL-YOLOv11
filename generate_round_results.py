from pathlib import Path
from ultralytics import YOLO
import pandas as pd


DATA = "datasets/merged/data.yaml"


MODELS = {

    "FedAvg": [
        "federated/server/converted/fedavg_round5.pt"
    ],

    "FedProx": [
        "federated/server/converted/fedprox_round5.pt"
    ]

}



results = []


for name, models in MODELS.items():

    for model_path in models:

        print("===================")
        print(name)
        print(model_path)
        print("===================")


        model = YOLO(model_path)


        metrics = model.val(
            data=DATA,
            imgsz=640,
            batch=8,
            device=0,
            workers=0
        )


        results.append({

            "Model": name,

            "Precision":
            metrics.box.mp,

            "Recall":
            metrics.box.mr,

            "mAP50":
            metrics.box.map50,

            "mAP50-95":
            metrics.box.map

        })



df = pd.DataFrame(results)


out = (
    "thesis_results/tables/"
    "statistics/final_statistical_results.csv"
)


df.to_csv(
    out,
    index=False
)


print(df)

print(
    "Saved:",
    out
)