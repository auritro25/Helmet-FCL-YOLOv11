import os
import glob
import torch

from torch.utils.data import DataLoader

from joint_cl.dataset import ContrastiveDataset
from joint_cl.yolo_cl_trainer import YOLOCLTrainer



def get_images(data_path):

    images = []

    for ext in [
        "*.jpg",
        "*.png",
        "*.jpeg"
    ]:

        images.extend(
            glob.glob(
                os.path.join(
                    data_path,
                    "images/train",
                    ext
                )
            )
        )

    return images



def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "Device:",
        device
    )


    image_root = "datasets/merged"


    images = get_images(
        image_root
    )


    print(
        "Images found:",
        len(images)
    )


    dataset = ContrastiveDataset(
        images
    )


    loader = DataLoader(

        dataset,

        batch_size=2,

        shuffle=True,

        num_workers=0,

        pin_memory=True

    )


    trainer = YOLOCLTrainer(

        model_path="yolo11s.pt",

        proj_dim=128,

        temperature=0.07

    )


    trainer.model.to(device)

    trainer.projector.to(device)


    optimizer = torch.optim.AdamW(

        list(trainer.model.parameters())
        +
        list(trainer.projector.parameters()),

        lr=1e-4

    )


    # -----------------------------
    # First forward test
    # -----------------------------

    print(
        "Testing first forward pass..."
    )


    test_batch = next(
        iter(loader)
    )


    test_view1, test_view2, _ = test_batch


    test_view1 = test_view1.to(
        device
    )


    with torch.no_grad():

        test_feature = trainer.forward_features(
            test_view1
        )


    print(
        "Feature test OK:",
        test_feature.shape
    )


    del test_batch
    del test_view1
    torch.cuda.empty_cache()



    epochs = 10



    for epoch in range(epochs):


        total_loss = 0


        for batch_idx, batch in enumerate(loader):


            if batch_idx % 10 == 0:

                print(
                    f"Epoch {epoch+1} "
                    f"Processing batch {batch_idx}/{len(loader)}"
                )


            view1, view2, label = batch


            view1 = view1.to(
                device
            )


            view2 = view2.to(
                device
            )


            optimizer.zero_grad()


            loss = trainer.training_step(

                (
                    view1,
                    view2,
                    label
                )

            )


            loss.backward()


            optimizer.step()


            total_loss += loss.item()



        avg_loss = (

            total_loss /
            len(loader)

        )


        print(

            f"Epoch {epoch+1}/{epochs} "
            f"Loss: {avg_loss:.4f}"

        )



    os.makedirs(
        "models",
        exist_ok=True
    )


    torch.save(

        {

            "model":
            trainer.model.state_dict(),

            "projector":
            trainer.projector.state_dict()

        },

        "models/contrastive_pretrained.pt"

    )


    print(
        "Contrastive pretraining completed"
    )



if __name__ == "__main__":

    main()