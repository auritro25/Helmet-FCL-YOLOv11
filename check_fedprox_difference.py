import torch


before = "models/phase2_supcon_best.pt"

after = "federated/server/global_fedprox_round_1.pt"


def load_weights(path):

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    if "model" in ckpt:

        model = ckpt["model"]

        if hasattr(model, "state_dict"):
            return model.state_dict()

        return model

    return ckpt



w1 = load_weights(before)
w2 = load_weights(after)



total_difference = 0
total_parameters = 0
changed_layers = 0


for key in w1:

    if key in w2:

        diff = torch.norm(
            w1[key] - w2[key]
        ).item()

        total_difference += diff

        total_parameters += w1[key].numel()

        if diff > 0:
            changed_layers += 1



average_difference = (
    total_difference /
    len(w1)
)



print("==============================")
print("FedProx Weight Difference")
print("==============================")

print(
    "Changed layers:",
    changed_layers,
    "/",
    len(w1)
)

print(
    "Average layer difference:",
    average_difference
)

print(
    "Total parameter difference:",
    total_difference
)

print(
    "Total parameters:",
    total_parameters
)