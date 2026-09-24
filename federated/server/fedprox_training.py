from pathlib import Path
import sys
import torch

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT)
)


from federated.fedprox.manual_fedprox import FedProxTrainer



torch.serialization.add_safe_globals(
    [DetectionModel]
)



CLIENTS = [

    "client1",
    "client2",
    "client3"

]


START_ROUND = 2

END_ROUND = 5


LOCAL_EPOCHS = 5


MU = 0.01



CLIENT_SIZES = {

    "client1": 1075,

    "client2": 2874,

    "client3": 1381

}



BASE_MODEL = (

    "models/phase2_supcon_best.pt"

)


# Existing global model from completed round 1

GLOBAL_MODEL = (

    "federated/server/global_fedprox_round_1.pt"

)





def aggregate(weights, sizes):


    total = sum(sizes)


    result = {}



    for key in weights[0]:


        result[key] = sum(

            w[key] * (s / total)

            for w, s in zip(

                weights,

                sizes

            )

        )


    return result





def save_global(weights, r):


    Path(

        "federated/server"

    ).mkdir(

        exist_ok=True

    )


    path = (

        f"federated/server/"
        f"global_fedprox_round_{r}.pt"

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





def load_global_model(path):


    model = YOLO(

        BASE_MODEL

    )



    ckpt = torch.load(

        path,

        map_location="cpu",

        weights_only=False

    )



    model.model.load_state_dict(

        ckpt["model"]

    )



    return model





def main():


    global_model_path = GLOBAL_MODEL



    for r in range(

        START_ROUND,

        END_ROUND + 1

    ):



        print()

        print("================")

        print(

            f"FEDPROX ROUND {r}"

        )

        print("================")





        base = load_global_model(

            global_model_path

        )



        global_weights = {


            k: v.detach().clone()


            for k,v in base.model.state_dict().items()


        }



        client_weights = []





        for client in CLIENTS:



            print()

            print(

                "Training",

                client

            )



            model = load_global_model(

                global_model_path

            )





            def add_fedprox(trainer):


                trainer.global_weights = {


                    k: v.detach().clone()


                    for k,v in global_weights.items()


                }



                trainer.mu = MU



                print(

                    "FedProx activated"

                )





            model.add_callback(

                "on_train_start",

                add_fedprox

            )





            model.train(

                trainer=FedProxTrainer,

                data=(

                    f"federated/clients/"
                    f"{client}/data.yaml"

                ),

                epochs=LOCAL_EPOCHS,

                imgsz=640,

                batch=8,

                device=0,

                workers=0,

                project="runs/fedprox_manual",

                name=f"{client}_round_{r}",

                exist_ok=True

            )



            client_weights.append(

                {

                    k: v.detach().clone()

                    for k,v in model.model.state_dict().items()

                }

            )





        sizes = [

            CLIENT_SIZES[c]

            for c in CLIENTS

        ]



        new_global = aggregate(

            client_weights,

            sizes

        )



        global_model_path = save_global(

            new_global,

            r

        )





    print()

    print(

        "FEDPROX RESUME COMPLETE"

    )





if __name__ == "__main__":

    main()