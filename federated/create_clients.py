from pathlib import Path
import shutil
import random
import numpy as np
import yaml


SOURCE = Path("datasets/merged")

OUTPUT = Path("federated/clients")


CLIENTS = [
    "client1",
    "client2",
    "client3"
]


# Dirichlet concentration parameter
# smaller value = stronger non-IID
ALPHA = 1.0


random.seed(42)
np.random.seed(42)



# Create client folders

for client in CLIENTS:

    (OUTPUT / client / "images/train").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT / client / "labels/train").mkdir(
        parents=True,
        exist_ok=True
    )



# Load images

images = list(
    (SOURCE / "images/train").glob("*.jpg")
)



# Collect image-level labels

image_labels = []


for img in images:

    label_file = (
        SOURCE /
        "labels/train" /
        (img.stem + ".txt")
    )


    classes = []


    if label_file.exists():

        with open(label_file) as f:

            for line in f:

                cls = int(
                    line.split()[0]
                )

                classes.append(cls)



    if len(classes) > 0:

        image_labels.append(
            (
                img,
                classes[0]
            )
        )



# Group images by class

class_images = {}


for img, cls in image_labels:

    if cls not in class_images:

        class_images[cls] = []


    class_images[cls].append(img)



# Store client images

client_images = {

    c: []

    for c in CLIENTS

}



# Dirichlet allocation

for cls, imgs in class_images.items():


    random.shuffle(imgs)


    proportions = np.random.dirichlet(
        [ALPHA] * len(CLIENTS)
    )


    counts = (
        proportions *
        len(imgs)
    ).astype(int)



    while counts.sum() < len(imgs):

        counts[
            np.argmax(proportions)
        ] += 1



    start = 0


    for i, client in enumerate(CLIENTS):


        end = start + counts[i]


        client_images[client].extend(
            imgs[start:end]
        )


        start = end





# Copy data

for client, imgs in client_images.items():


    print(
        client,
        "images:",
        len(imgs)
    )


    for img in imgs:


        label = (

            SOURCE /
            "labels/train" /
            (img.stem + ".txt")

        )


        shutil.copy(

            img,

            OUTPUT /
            client /
            "images/train" /
            img.name

        )


        if label.exists():

            shutil.copy(

                label,

                OUTPUT /
                client /
                "labels/train" /
                label.name

            )





# Create YAML files

for client in CLIENTS:


    yaml_path = (
        OUTPUT /
        client /
        "data.yaml"
    )


    data_yaml = {


        "path": str(
            (OUTPUT / client).resolve()
        ),


        "train": "images/train",


        "val": str(
            (SOURCE / "images/test").resolve()
        ),


        "names": {

            0: "rider",

            1: "no_helmet",

            2: "helmet"

        }

    }



    with open(
        yaml_path,
        "w"
    ) as f:


        yaml.dump(

            data_yaml,

            f,

            sort_keys=False

        )


    print(
        client,
        "data.yaml created"
    )





print(
    "\nDirichlet client datasets created successfully"
)