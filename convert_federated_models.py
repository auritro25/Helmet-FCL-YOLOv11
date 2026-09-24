import torch
from pathlib import Path

from ultralytics import YOLO


BASE_MODEL = "models/phase2_supcon_best.pt"


MODELS = {

    "FedAvg":
    "federated/server/global_fedavg_round_5.pt",

    "FedProx":
    "federated/server/global_fedprox_round_5.pt"

}


OUTPUT = Path(
    "federated/server/converted"
)

OUTPUT.mkdir(
    exist_ok=True
)



for name,path in MODELS.items():

    print()
    print("===================")
    print(name)
    print("===================")


    # load original YOLO model structure
    yolo = YOLO(
        BASE_MODEL
    )


    # load federated weights

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )


    weights = checkpoint["model"]


    # apply weights

    yolo.model.load_state_dict(
        weights
    )


    output = (
        OUTPUT /
        f"{name.lower()}_round5.pt"
    )


    yolo.save(
        output
    )


    print(
        "Saved:",
        output
    )


print()
print("Conversion complete")