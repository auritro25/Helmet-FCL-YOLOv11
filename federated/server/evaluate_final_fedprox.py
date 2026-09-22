from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel


torch.serialization.add_safe_globals(
    [DetectionModel]
)


BASE_MODEL = "models/phase2_supcon_best.pt"


FEDPROX_MODEL = (
    "federated/server/global_fedprox_round_1.pt"
)



def main():

    model = YOLO(
        BASE_MODEL
    )


    checkpoint = torch.load(
        FEDPROX_MODEL,
        map_location="cpu",
        weights_only=False
    )


    model.model.load_state_dict(
        checkpoint["model"],
        strict=False
    )


    print(
        "FedProx Round 1 model loaded"
    )


    results = model.val(

        data="datasets/merged/data.yaml",

        split="test",

        imgsz=640,

        batch=8,

        workers=0,

        device=0,

        project="runs/final_results",

        name="fedprox_round1"

    )


    print(
        "\nFINAL FEDPROX RESULTS"
    )

    print(
        results.results_dict
    )



if __name__ == "__main__":

    main()