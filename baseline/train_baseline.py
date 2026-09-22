from ultralytics import YOLO


def main():

    model = YOLO("yolo11s.pt")

    model.train(

        data="datasets/merged/data.yaml",

        epochs=25,

        imgsz=640,

        batch=8,

        optimizer="AdamW",

        lr0=0.001,

        patience=20,

        amp=True,

        device=0,

        workers=0,

        project="runs/baseline",

        name="yolo11s_baseline"

    )


if __name__ == "__main__":
    main()