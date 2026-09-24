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


SEEDS = [42, 7, 123]


OUTPUT = Path(
    "thesis_results/statistics"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)



def run():

    results = []


    for name, path in MODELS.items():

        print("\n====================")
        print(name)
        print("====================")


        model = YOLO(path)


        for seed in SEEDS:

            print(
                "Seed:",
                seed
            )


            metrics = model.val(

                data=DATA,

                split="val",

                seed=seed,

                workers=0,

                batch=8,

                verbose=False

            )


            precision = metrics.box.mp * 100
            recall = metrics.box.mr * 100
            map50 = metrics.box.map50 * 100
            map5095 = metrics.box.map * 100


            print(
                f"P={precision:.3f}",
                f"R={recall:.3f}",
                f"mAP50={map50:.3f}",
                f"mAP50-95={map5095:.3f}"
            )


            results.append(

                [
                    name,
                    seed,
                    precision,
                    recall,
                    map50,
                    map5095
                ]

            )



    df = pd.DataFrame(

        results,

        columns=[

            "Model",
            "Seed",
            "Precision",
            "Recall",
            "mAP50",
            "mAP50-95"

        ]

    )


    df.to_csv(

        OUTPUT/"seed_results.csv",

        index=False

    )


    summary = df.groupby(
        "Model"
    ).agg(
        {
            "Precision":["mean","std"],
            "Recall":["mean","std"],
            "mAP50":["mean","std"],
            "mAP50-95":["mean","std"]
        }
    )


    summary.to_csv(

        OUTPUT/"multiseed_summary.csv"

    )


    print("\nDONE")
    print(summary)



if __name__ == "__main__":

    run()