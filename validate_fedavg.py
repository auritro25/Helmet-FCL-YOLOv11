import torch
from ultralytics import YOLO


BASE_MODEL = "models/phase2_supcon_best.pt"


GLOBAL_MODEL = (
    "federated/server/global_fedavg_round_5.pt"
)


DATA = (
    "datasets/merged/data.yaml"
)



def main():


    model = YOLO(

        BASE_MODEL

    )


    ckpt = torch.load(

        GLOBAL_MODEL,

        map_location="cpu",

        weights_only=False

    )


    model.model.load_state_dict(

        ckpt["model"]

    )



    print("==============================")
    print("FEDAVG FINAL GLOBAL MODEL")
    print("==============================")



    model.val(

        data=DATA,

        imgsz=640,

        batch=8,

        device=0,

        workers=0

    )



    print("==============================")
    print("FedAvg Validation Complete")
    print("==============================")



if __name__ == "__main__":

    main()