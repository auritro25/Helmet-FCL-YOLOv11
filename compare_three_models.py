from ultralytics import YOLO
import torch


DATA = "datasets/merged/data.yaml"


MODELS = {

    "Centralized YOLOv11":
    "models/phase2_supcon_best.pt",

    "FedAvg":
    "federated/server/global_fedavg_round_5.pt",

    "FedProx":
    "federated/server/global_fedprox_round_5.pt"

}



def load_federated_model(path):

    # create YOLO architecture first
    model = YOLO(
        "models/phase2_supcon_best.pt"
    )


    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )


    # load federated weights
    model.model.load_state_dict(
        checkpoint["model"]
    )


    return model





results = {}



for name, path in MODELS.items():


    print("\n")
    print("="*60)
    print(name)
    print("="*60)



    if "FedAvg" in name or "FedProx" in name:

        model = load_federated_model(path)

    else:

        model = YOLO(path)



    metrics = model.val(

        data=DATA,

        imgsz=640,

        batch=8,

        device=0,

        workers=0

    )



    results[name] = {

        "Precision":
        metrics.box.mp,

        "Recall":
        metrics.box.mr,

        "mAP50":
        metrics.box.map50,

        "mAP50-95":
        metrics.box.map

    }




print("\n\n")

print("="*80)

print("FINAL THREE MODEL COMPARISON")

print("="*80)


print(
    f"{'Model':25}"
    f"{'Precision':12}"
    f"{'Recall':12}"
    f"{'mAP50':12}"
    f"{'mAP50-95':12}"
)


print("-"*80)



for name, r in results.items():

    print(

        f"{name:25}"

        f"{r['Precision']:.3f}       "

        f"{r['Recall']:.3f}       "

        f"{r['mAP50']:.3f}       "

        f"{r['mAP50-95']:.3f}"

    )


print("="*80)