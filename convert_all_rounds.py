from pathlib import Path
import torch
from ultralytics import YOLO


BASE_MODEL = "models/phase2_supcon_best.pt"


OUT = Path(
    "federated/server/converted"
)

OUT.mkdir(
    exist_ok=True
)


models = []


for method in ["fedavg", "fedprox"]:

    for r in range(1,6):

        models.append(
            (
                method,
                r,
                f"federated/server/global_{method}_round_{r}.pt"
            )
        )



for method, r, path in models:

    print("================")
    print(method, "Round", r)
    print("================")


    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )


    weights = checkpoint["model"]


    model = YOLO(
        BASE_MODEL
    )


    model.model.load_state_dict(
        weights
    )


    save_path = (
        OUT /
        f"{method}_round{r}.pt"
    )


    model.save(
        save_path
    )


    print(
        "Saved:",
        save_path
    )


print("ALL CONVERTED")