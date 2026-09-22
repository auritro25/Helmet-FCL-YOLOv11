from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt


ROOT = Path("datasets/merged")

classes = {
    0: "rider",
    1: "no_helmet",
    2: "helmet"
}


splits = ["train", "val", "test"]


for split in splits:

    label_path = ROOT / "labels" / split

    counter = Counter()
    images = 0
    annotations = 0

    for txt in label_path.glob("*.txt"):

        images += 1

        with open(txt) as f:
            lines = f.readlines()

        annotations += len(lines)

        for line in lines:
            cls = int(line.split()[0])
            counter[classes[cls]] += 1


    print("\n====================")
    print(split.upper())
    print("====================")

    print("Images:", images)
    print("Annotations:", annotations)

    for c, n in counter.items():
        print(c, ":", n)


    plt.figure(figsize=(6,4))

    plt.bar(
        counter.keys(),
        counter.values()
    )

    plt.title(
        f"{split} Class Distribution"
    )

    plt.ylabel(
        "Number of Objects"
    )

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        f"{split}_distribution.png"
    )

    plt.close()