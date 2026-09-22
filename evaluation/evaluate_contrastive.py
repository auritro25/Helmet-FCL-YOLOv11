import torch
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import cv2
import numpy as np

import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)


from contrastive.model_cl import YOLOContrastive


DATASET = Path("datasets/merged")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 640
BATCH_SIZE = 16


class HelmetDataset(torch.utils.data.Dataset):

    def __init__(self):

        self.images = list(
            (DATASET/"images"/"test").glob("*")
        )

        self.labels = DATASET/"labels"/"test"


    def __len__(self):
        return len(self.images)


    def __getitem__(self,index):

        img_path=self.images[index]

        img=cv2.imread(str(img_path))

        img=cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        img=cv2.resize(
            img,
            (IMAGE_SIZE,IMAGE_SIZE)
        )

        img=img/255.0

        img=torch.tensor(
            img,
            dtype=torch.float32
        ).permute(2,0,1)


        label_path=self.labels/(img_path.stem+".txt")

        cls=0

        if label_path.exists():

            line=open(label_path).readline()

            if line:
                cls=int(line.split()[0])


        return img,cls



def main():

    dataset=HelmetDataset()

    loader=DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


    model=YOLOContrastive(
        "yolo11s.pt"
    )


    checkpoint=torch.load(
        "yolo11s_contrastive.pt",
        map_location=DEVICE
    )


    model.load_state_dict(
        checkpoint
    )


    model.to(DEVICE)

    model.eval()


    embeddings=[]
    labels=[]


    with torch.no_grad():

        for images,cls in loader:

            images=images.to(DEVICE)

            emb=model(images)

            embeddings.extend(
                emb.cpu().numpy()
            )

            labels.extend(
                cls.numpy()
            )


    embeddings=np.array(
        embeddings
    )


    print(
        "Embeddings:",
        embeddings.shape
    )


    tsne=TSNE(
        n_components=2,
        random_state=42
    )


    reduced=tsne.fit_transform(
        embeddings
    )


    plt.figure(figsize=(8,6))

    for c in range(3):

        idx=np.array(labels)==c

        plt.scatter(
            reduced[idx,0],
            reduced[idx,1],
            label=str(c)
        )


    plt.legend()

    plt.title(
        "YOLOv11 + Contrastive Learning Feature Space"
    )

    plt.savefig(
        "contrastive_tsne.png",
        dpi=300
    )


    print(
        "Saved contrastive_tsne.png"
    )



if __name__=="__main__":
    main()