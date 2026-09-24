from pathlib import Path
from ultralytics import YOLO
import random
import shutil


MODELS = {

    "centralized":
    "models/phase2_supcon_best.pt",

    "fedavg":
    "federated/server/converted/fedavg_round5.pt",

    "fedprox":
    "federated/server/converted/fedprox_round5.pt"

}


IMAGE_DIR = Path(
    "datasets/merged/images/val"
)


OUTPUT = Path(
    "thesis_results/detection_examples"
)


OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


random.seed(42)


images = list(
    IMAGE_DIR.glob("*.jpg")
)


selected = random.sample(
    images,
    10
)


print("\nSelected:")
for x in selected:
    print(x.name)



for name, path in MODELS.items():

    print()
    print("================")
    print(name)
    print("================")


    model = YOLO(path)


    output_folder = OUTPUT / name

    output_folder.mkdir(
        exist_ok=True
    )


    for img in selected:


        results = model.predict(
            source=str(img),
            imgsz=640,
            conf=0.25,
            save=True,
            project=str(output_folder),
            name="predict",
            exist_ok=True,
            verbose=False
        )


        save_location = Path(
            results[0].save_dir
        )


        generated_image = (
            save_location /
            img.name
        )


        if generated_image.exists():

            shutil.copy2(
                generated_image,
                output_folder / img.name
            )

            print(
                "saved:",
                name,
                img.name
            )



print()
print("==========================")
print("DETECTION EXAMPLES DONE")
print("==========================")