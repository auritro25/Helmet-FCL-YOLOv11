from ultralytics import YOLO
import torch
from pathlib import Path

from ultralytics.nn.tasks import DetectionModel


torch.serialization.add_safe_globals(
    [DetectionModel]
)


BASE_MODEL = "models/phase2_supcon_best.pt"


CLIENTS = [
    "client1",
    "client2",
    "client3"
]


ROUNDS = 5
LOCAL_EPOCHS = 5



# Number of training images per client
# Generated from Dirichlet split
CLIENT_SIZES = {

    "client1": 1075,

    "client2": 2874,

    "client3": 1381

}



def train_client(
        client,
        model_path,
        round_no
):


    print(
        f"\nTraining {client} - Round {round_no}"
    )


    model = YOLO(
        model_path
    )


    model.train(

        data=(
            f"federated/clients/"
            f"{client}/data.yaml"
        ),

        epochs=LOCAL_EPOCHS,

        imgsz=640,

        batch=8,

        device=0,

        workers=0,

        project="runs/fl_final",

        name=f"{client}_round_{round_no}",

        exist_ok=True

    )


    return (

        f"runs/detect/runs/fl_final/"
        f"{client}_round_{round_no}/weights/best.pt"

    )



def load_weights(path):


    checkpoint = torch.load(

        path,

        map_location="cpu",

        weights_only=False

    )


    return checkpoint["model"].state_dict()



# Weighted FedAvg

def fedavg(
        weight_list,
        client_sizes
):


    avg = {}


    total_samples = sum(
        client_sizes
    )


    for key in weight_list[0]:


        weighted_sum = 0


        for weights, size in zip(
            weight_list,
            client_sizes
        ):


            weighted_sum += (
                weights[key]
                *
                (size / total_samples)
            )



        avg[key] = weighted_sum



    return avg




def save_checkpoint(
        state_dict,
        round_no
):


    save_path = (

        f"federated/server/"
        f"global_fedavg_round_{round_no}.pt"

    )


    torch.save(

        {
            "model": state_dict
        },

        save_path

    )


    print(
        "Saved:",
        save_path
    )


    return save_path




def convert_to_yolo_checkpoint(
        state_dict,
        round_no
):


    base = YOLO(
        BASE_MODEL
    )


    base.model.load_state_dict(
        state_dict
    )


    path = (

        f"federated/server/"
        f"yolo_global_round_{round_no}.pt"

    )


    base.save(
        path
    )


    return path




def main():


    global_model = BASE_MODEL



    for r in range(
        1,
        ROUNDS + 1
    ):


        print(
            "\n======================"
        )

        print(
            f"FEDAVG ROUND {r}"
        )

        print(
            "======================"
        )



        client_models = []



        for client in CLIENTS:


            result = train_client(

                client,

                global_model,

                r

            )


            client_models.append(
                result
            )




        weights = []



        for model in client_models:


            weights.append(
                load_weights(model)
            )



        sizes = [

            CLIENT_SIZES[c]

            for c in CLIENTS

        ]



        global_weights = fedavg(

            weights,

            sizes

        )



        save_checkpoint(

            global_weights,

            r

        )



        global_model = convert_to_yolo_checkpoint(

            global_weights,

            r

        )





if __name__ == "__main__":

    main()