from pathlib import Path
from ultralytics import YOLO


MODELS = {

    "Centralized YOLOv11":
    "models/phase2_supcon_best.pt",

    "FedAvg":
    "federated/server/converted/fedavg_round5.pt",

    "FedProx":
    "federated/server/converted/fedprox_round5.pt"

}



DATA = "datasets/merged/data.yaml"



OUTPUT = Path(
    "thesis_results/graphs/confusion_matrix"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)



for name, model_path in MODELS.items():

    print()
    print("====================")
    print(name)
    print("====================")


    model = YOLO(model_path)


    save_dir = OUTPUT / name.replace(" ","_")


    model.val(

        data=DATA,

        imgsz=640,

        batch=8,

        workers=0,

        plots=True,

        project=str(save_dir),

        name=""

    )


    print(
        "Finished:",
        name
    )


print()
print("DONE")