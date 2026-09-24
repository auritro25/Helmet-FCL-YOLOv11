from ultralytics import YOLO
from pathlib import Path
import shutil


MODEL = r"federated\server\converted\fedprox_round5.pt"


IMAGE_DIR = Path(
    "datasets/merged/images/val"
)


OUTPUT = Path(
    "thesis_results/detection_examples/failure_cases"
)


OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


model = YOLO(MODEL)


low_conf = OUTPUT / "low_confidence"

low_conf.mkdir(
    exist_ok=True
)


count = 0


for img in IMAGE_DIR.glob("*.jpg"):


    results = model.predict(

        source=str(img),

        conf=0.10,

        verbose=False

    )


    detections = results[0].boxes


    if len(detections) > 0:


        confidences = detections.conf.cpu().numpy()


        # save uncertain predictions

        if confidences.min() < 0.35:


            shutil.copy(

                img,

                low_conf / img.name

            )


            count += 1



print(
    "Low confidence cases:",
    count
)


print("DONE")