import torch
from ultralytics.nn.tasks import DetectionModel


# allow Ultralytics model loading in PyTorch 2.6
torch.serialization.add_safe_globals(
    [DetectionModel]
)


def get_weights(path):

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    return checkpoint["model"].state_dict()



def average_weights(models):

    avg_state = {}

    for key in models[0].keys():

        avg_state[key] = sum(
            model[key]
            for model in models
        ) / len(models)

    return avg_state



client_models = [

    "runs/detect/runs/federated/client1/weights/best.pt",

    "runs/detect/runs/federated/client2/weights/best.pt",

    "runs/detect/runs/federated/client3/weights/best.pt"

]


weights = []


for path in client_models:

    print(
        "Loading:",
        path
    )

    weights.append(
        get_weights(path)
    )



global_weights = average_weights(
    weights
)



output = (
    "federated/server/"
    "global_fedavg_round1.pt"
)


torch.save(
    {
        "model": global_weights
    },
    output
)


print()
print("FedAvg completed")
print("Saved:", output)