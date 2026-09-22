import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import cv2
import numpy as np

from model_cl import YOLOContrastive
from contrastive_loss import SupConLoss


# ==========================
# Configuration
# ==========================

DATASET = Path("datasets/merged")

IMAGE_SIZE = 640
BATCH_SIZE = 8
EPOCHS = 25
LR = 1e-4


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================
# Dataset
# ==========================

class HelmetDataset(Dataset):

    def __init__(self, split="train"):

        self.images = (
            DATASET /
            "images" /
            split
        )

        self.labels = (
            DATASET /
            "labels" /
            split
        )


        self.files = list(
            self.images.glob("*")
        )


    def __len__(self):

        return len(self.files)


    def __getitem__(self, idx):

        img_path = self.files[idx]


        img = cv2.imread(
            str(img_path)
        )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )


        img = cv2.resize(
            img,
            (IMAGE_SIZE, IMAGE_SIZE)
        )


        img = img / 255.0


        img = torch.tensor(
            img,
            dtype=torch.float32
        )


        img = img.permute(
            2,0,1
        )


        # read class label
        label_path = (
            self.labels /
            (img_path.stem + ".txt")
        )


        classes=[]


        if label_path.exists():

            for line in open(label_path):

                cls=int(
                    line.split()[0]
                )

                classes.append(cls)


        # image level label
        # use first object class

        if len(classes)>0:

            label=classes[0]

        else:

            label=0


        return img,label



# ==========================
# Training
# ==========================


def main():

    print(
        "Using:",
        DEVICE
    )


    dataset = HelmetDataset(
        "train"
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )


    model = YOLOContrastive(
        "yolo11s.pt"
    )


    model.to(
        DEVICE
    )


    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=LR
    )


    criterion=SupConLoss()


    model.train()


    for epoch in range(EPOCHS):

        total_loss=0


        for images,labels in loader:


            images=images.to(
                DEVICE
            )

            labels=labels.to(
                DEVICE
            )


            embeddings=model(
                images
            )


            loss=criterion(
                embeddings,
                labels
            )


            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            total_loss += loss.item()



        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {total_loss/len(loader):.4f}"
        )


    torch.save(
        model.state_dict(),
        "yolo11s_contrastive.pt"
    )


    print(
        "Training completed"
    )



if __name__=="__main__":

    main()