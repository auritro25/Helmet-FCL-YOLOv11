from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel


torch.serialization.add_safe_globals(
    [DetectionModel]
)


BASE_MODEL = "models/phase2_supcon_best.pt"

FEDAVG_MODEL = (
    "federated/server/global_fedavg_round_10.pt"
)


def main():

    # Load YOLO architecture
    model = YOLO(
        BASE_MODEL
    )


    # Load FedAvg weights
    checkpoint = torch.load(
        FEDAVG_MODEL,
        map_location="cpu",
        weights_only=False
    )


    model.model.load_state_dict(
        checkpoint["model"],
        strict=False
    )


    print(
        "FedAvg Round 10 model loaded"
    )


    results = model.val(

        data="datasets/merged/data.yaml",

        split="test",

        imgsz=640,

        batch=8,

        workers=0,

        device=0,

        project="runs/final_results",

        name="fedavg_round10"

    )


    print("\nFINAL FEDAVG RESULTS")
    print(results.results_dict)



if __name__ == "__main__":

    main()