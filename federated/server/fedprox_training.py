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


ROUNDS = 5


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





def aggregate(

        weights,

        sizes

):


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





def save_global(

        weights,

        r

):


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





def main():


    global_model = BASE_MODEL



    for r in range(

        1,

        ROUNDS + 1

    ):



        print()

        print("================")

        print(

            f"FEDPROX ROUND {r}"

        )

        print("================")



        base = YOLO(

            global_model

        )



        global_weights = {


            k: v.detach().clone()


            for k, v in base.model.state_dict().items()


        }



        client_weights = []





        for client in CLIENTS:



            print()

            print(

                "Training",

                client

            )



            model = YOLO(

                global_model

            )





            def add_fedprox(trainer):


                trainer.global_weights = {


                    k: v.detach().clone()


                    for k, v in global_weights.items()


                }



                trainer.mu = MU



                print(

                    "FedProx enabled:",

                    trainer.mu

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

                    for k, v in model.model.state_dict().items()

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





        global_model = save_global(

            new_global,

            r

        )





    print()

    print(

        "FEDPROX COMPLETE"

    )





if __name__ == "__main__":

    main()