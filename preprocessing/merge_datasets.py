import os
import shutil
from pathlib import Path
import random


# ==============================
# Paths
# ==============================

ROOT = Path(__file__).resolve().parents[1]

ROBOFLOW = ROOT / "datasets" / "roboflow"
EDGEVISION = ROOT / "datasets" / "edgevision"

OUTPUT = ROOT / "datasets" / "merged"


# Final classes
# 0 rider
# 1 no_helmet
# 2 helmet

CLASS_NAMES = [
    "rider",
    "no_helmet",
    "helmet"
]


def create_folders():

    for split in ["train", "val", "test"]:

        (OUTPUT / "images" / split).mkdir(
            parents=True,
            exist_ok=True
        )

        (OUTPUT / "labels" / split).mkdir(
            parents=True,
            exist_ok=True
        )


def copy_data(
        image,
        label,
        split,
        new_name
):

    shutil.copy(
        image,
        OUTPUT /
        "images" /
        split /
        new_name
    )

    shutil.copy(
        label,
        OUTPUT /
        "labels" /
        split /
        new_name.replace(
            image.suffix,
            ".txt"
        )
    )


# -----------------------------
# Roboflow processing
# -----------------------------

def process_roboflow():

    print("Processing Roboflow...")


    mapping = {
        0:2,   # With Helmet
        1:1    # Without Helmet
    }


    for split in ["train","valid","test"]:

        out_split = (
            "val"
            if split=="valid"
            else split
        )


        img_dir = ROBOFLOW / split / "images"
        lbl_dir = ROBOFLOW / split / "labels"


        for img in img_dir.iterdir():

            label = lbl_dir / (
                img.stem + ".txt"
            )

            if not label.exists():
                continue


            new_label=[]


            for line in open(label):

                parts=line.strip().split()

                cls=int(parts[0])

                parts[0]=str(
                    mapping[cls]
                )

                new_label.append(
                    " ".join(parts)
                )


            new_name = (
                "roboflow_"
                + img.name
            )


            shutil.copy(
                img,
                OUTPUT/
                "images"/
                out_split/
                new_name
            )


            with open(
                OUTPUT/
                "labels"/
                out_split/
                new_name.replace(
                    img.suffix,
                    ".txt"
                ),
                "w"
            ) as f:

                f.write(
                    "\n".join(new_label)
                )



# -----------------------------
# EdgeVision processing
# -----------------------------

def process_edgevision():

    print("Processing EdgeVision...")


    mapping={
        0:0, # BikeWithRider
        1:1, # NoHelmet
        2:2  # Helmet
    }


    img_dir = EDGEVISION/"images"

    lbl_dir = (
        EDGEVISION/
        "labels"/
        "yolo"
    )


    images=list(img_dir.iterdir())

    random.shuffle(images)


    for i,img in enumerate(images):

        label=lbl_dir/(img.stem+".txt")


        if not label.exists():
            continue


        split="train"

        if i < len(images)*0.15:
            split="val"

        elif i < len(images)*0.20:
            split="test"


        output=[]


        for line in open(label):

            parts=line.strip().split()

            cls=int(parts[0])

            parts[0]=str(
                mapping[cls]
            )

            output.append(
                " ".join(parts)
            )


        new_name="edgevision_"+img.name


        shutil.copy(
            img,
            OUTPUT/
            "images"/
            split/
            new_name
        )


        with open(
            OUTPUT/
            "labels"/
            split/
            new_name.replace(
                img.suffix,
                ".txt"
            ),
            "w"
        ) as f:

            f.write(
                "\n".join(output)
            )


def create_yaml():

    yaml="""

path: datasets/merged

train: images/train
val: images/val
test: images/test


names:

  0: rider
  1: no_helmet
  2: helmet

"""


    with open(
        OUTPUT/"data.yaml",
        "w"
    ) as f:

        f.write(yaml)



if __name__=="__main__":

    create_folders()

    process_roboflow()

    process_edgevision()

    create_yaml()


    print(
        "Dataset merging completed!"
    )