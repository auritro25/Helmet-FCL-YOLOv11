from pathlib import Path
import shutil
import random


SOURCE = Path("datasets/merged")

OUTPUT = Path("federated/clients")


clients = {
    "client1":0.33,
    "client2":0.33,
    "client3":0.34
}


for c in clients:

    (OUTPUT/c/"images/train").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT/c/"labels/train").mkdir(
        parents=True,
        exist_ok=True
    )


images = list(
    (SOURCE/"images/train").glob("*.jpg")
)

random.shuffle(images)


n=len(images)

splits=[
    images[:n//3],
    images[n//3:2*n//3],
    images[2*n//3:]
]


for idx, data in enumerate(splits):

    client=f"client{idx+1}"

    for img in data:

        label=(
            SOURCE/
            "labels/train"/
            (img.stem+".txt")
        )


        shutil.copy(
            img,
            OUTPUT/client/"images/train"/img.name
        )


        if label.exists():

            shutil.copy(
                label,
                OUTPUT/client/"labels/train"/label.name
            )


print("Client datasets created")