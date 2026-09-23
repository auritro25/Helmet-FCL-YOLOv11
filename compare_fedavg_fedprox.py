import torch


fedavg_path = "federated/server/global_fedavg_round_1.pt"

fedprox_path = "federated/server/global_fedprox_round_1.pt"



def load_weights(path):

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    model = ckpt["model"]

    if hasattr(model, "state_dict"):
        return model.state_dict()

    return model



fedavg = load_weights(fedavg_path)

fedprox = load_weights(fedprox_path)



total_difference = 0
changed_layers = 0
total_layers = 0


max_difference = 0
max_layer = ""


for key in fedavg:

    if key in fedprox:

        diff = torch.norm(
            fedavg[key] - fedprox[key]
        ).item()


        total_difference += diff

        total_layers += 1


        if diff > 0:
            changed_layers += 1


        if diff > max_difference:
            max_difference = diff
            max_layer = key



print("==============================")
print("FedAvg vs FedProx Round 1")
print("==============================")

print(
    "Changed layers:",
    changed_layers,
    "/",
    total_layers
)

print(
    "Average difference:",
    total_difference / total_layers
)

print(
    "Total difference:",
    total_difference
)

print(
    "Largest changed layer:",
    max_layer
)

print(
    "Largest difference:",
    max_difference
)