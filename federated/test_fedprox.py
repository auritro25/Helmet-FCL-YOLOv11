from clients.fedprox_trainer import train_fedprox


model = "models/phase2_supcon_best.pt"


train_fedprox(
    "client3",
    model,
    epochs=5,
    mu=0.01
)