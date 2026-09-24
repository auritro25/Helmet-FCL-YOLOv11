from ultralytics import YOLO
from pathlib import Path


models = {

    "Centralized YOLOv11":
    r"runs\detect\runs\baseline\yolo11s_baseline-4\weights\best.pt",

    "FedAvg":
    r"federated\server\converted\fedavg_round5.pt",

    "FedProx":
    r"federated\server\converted\fedprox_round5.pt"

}


data_yaml = "datasets/merged/data.yaml"


output = Path(
    "thesis_results/graphs/pr_curve"
)

output.mkdir(
    parents=True,
    exist_ok=True
)



def generate():

    for name, path in models.items():

        print("\n================")
        print(name)
        print("================")


        model = YOLO(path)


        results = model.val(

            data=data_yaml,

            split="val",

            imgsz=640,

            batch=8,

            workers=0,

            plots=True

        )


        save_dir = Path(
            results.save_dir
        )


        for file in save_dir.glob("*PR_curve.png"):

            new_name = (
                name
                .replace(" ", "_")
                + "_PR_curve.png"
            )


            file.rename(
                output / new_name
            )


        print(
            "Saved:",
            name
        )



if __name__ == "__main__":

    generate()


    print("\nDONE")