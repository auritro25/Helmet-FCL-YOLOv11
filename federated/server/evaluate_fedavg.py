from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel


torch.serialization.add_safe_globals(
    [DetectionModel]
)


def main():

    # Load 3-class YOLO architecture
    model = YOLO(
        "models/phase2_supcon_best.pt"
    )


    # Load FedAvg weights
    checkpoint = torch.load(
        "federated/server/global_fedavg_round1.pt",
        map_location="cpu",
        weights_only=False
    )


    model.model.load_state_dict(
        checkpoint["model"],
        strict=False
    )


    print("FedAvg weights loaded")


    results = model.val(

        data="datasets/merged/data.yaml",

        split="test",

        imgsz=640,

        batch=8,

        workers=0,

        device=0,

        project="runs/federated_eval",

        name="fedavg_round1"

    )


    print(results)


if __name__ == "__main__":

    main()