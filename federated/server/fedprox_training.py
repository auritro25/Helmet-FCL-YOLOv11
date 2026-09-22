from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel


torch.serialization.add_safe_globals(
    [DetectionModel]
)


CLIENTS = [
    "client1",
    "client2",
    "client3"
]


ROUNDS = 10
LOCAL_EPOCHS = 5

MU = 0.01



def load_weights(path):

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    return ckpt["model"].state_dict()



def fedprox_aggregate(
    local_weights,
    global_weights
):

    new_weights = {}


    for key in local_weights[0]:

        avg = sum(
            w[key]
            for w in local_weights
        ) / len(local_weights)


        # FedProx correction
        if global_weights is not None:

            avg = avg + MU * (
                global_weights[key] - avg
            )


        new_weights[key] = avg


    return new_weights



def save_checkpoint(
    weights,
    round_no
):

    path = (
        f"federated/server/"
        f"global_fedprox_round_{round_no}.pt"
    )


    torch.save(
        {
            "model": weights
        },
        path
    )


    print(
        "Saved:",
        path
    )


    return path



def main():


    # Continue from completed Round 1
    global_model = (
        "federated/server/"
        "yolo_fedprox_round_1.pt"
    )


    for r in range(
        2,
        ROUNDS + 1
    ):


        print(
            "\n================"
        )

        print(
            f"FEDPROX ROUND {r}"
        )

        print(
            "================"
        )


        # Load previous global weights

        ckpt = torch.load(
            global_model,
            map_location="cpu",
            weights_only=False
        )


        if hasattr(
            ckpt["model"],
            "state_dict"
        ):

            global_weights = (
                ckpt["model"]
                .state_dict()
            )

        else:

            global_weights = ckpt["model"]



        client_models = []


        for client in CLIENTS:


            print(
                f"\nTraining {client}"
            )


            model = YOLO(
                global_model
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

                project="runs/fedprox",

                name=f"{client}_round_{r}",

                exist_ok=True

            )


            client_models.append(

                f"runs/detect/runs/fedprox/"
                f"{client}_round_{r}/weights/best.pt"

            )



        local_weights = []


        for path in client_models:

            local_weights.append(
                load_weights(path)
            )



        new_weights = fedprox_aggregate(

            local_weights,

            global_weights

        )


        global_model = save_checkpoint(

            new_weights,

            r

        )



if __name__ == "__main__":

    main()