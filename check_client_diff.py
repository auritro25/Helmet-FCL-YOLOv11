import torch


client1 = torch.load(
    "runs/detect/runs/fedprox/client1_round_1/weights/best.pt",
    map_location="cpu",
    weights_only=False
)

client2 = torch.load(
    "runs/detect/runs/fedprox/client2_round_1/weights/best.pt",
    map_location="cpu",
    weights_only=False
)


w1 = client1["model"].state_dict()
w2 = client2["model"].state_dict()


changed = 0
total_difference = 0


for key in w1:

    diff = torch.sum(
        torch.abs(
            w1[key] - w2[key]
        )
    ).item()


    if diff > 0:
        changed += 1


    total_difference += diff



print("==============================")
print("Client1 vs Client2")
print("==============================")
print("Changed layers:", changed)
print("Total difference:", total_difference)