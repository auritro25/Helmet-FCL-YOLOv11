from ultralytics import YOLO
from pathlib import Path
import shutil


MODELS = {

    "Centralized":
    r"runs\detect\runs\baseline\yolo11s_baseline-6\weights\best.pt",

    "FedAvg":
    r"federated\server\converted\fedavg_round5.pt",

    "FedProx":
    r"federated\server\converted\fedprox_round5.pt"

}



SOURCE = (
    "datasets/merged/images/val"
)


OUTPUT = Path(
    "thesis_results/detection_examples/nohelmet_analysis"
)


OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)



for name, model_path in MODELS.items():

    print("\n================")
    print(name)
    print("================")


    model = YOLO(
        model_path
    )


    save_dir = OUTPUT / name


    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    results = model.predict(

        source=SOURCE,

        conf=0.25,

        save=True,

        project=str(save_dir),

        name="predictions",

        exist_ok=True,

        workers=0

    )


    print(
        "Completed:",
        name
    )


print("\nNO HELMET ANALYSIS COMPLETE")