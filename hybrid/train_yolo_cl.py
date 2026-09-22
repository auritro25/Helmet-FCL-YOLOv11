from ultralytics import YOLO


def main():

    model = YOLO("yolo11s.pt")

    model.train(

        data="datasets/merged/data.yaml",

        epochs=25,

        imgsz=640,

        batch=8,

        device=0,

        optimizer="AdamW",

        lr0=0.001,

        amp=True,

        patience=20,

        workers=0,

        project="runs/hybrid",

        name="yolo11s_supcon"

    )


if __name__ == "__main__":
    main()