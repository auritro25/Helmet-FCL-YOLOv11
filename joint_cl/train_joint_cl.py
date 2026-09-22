from ultralytics import YOLO

from yolo_cl_trainer import YOLOCLTrainer


def main():

    model = YOLO(
        "yolo11s.pt"
    )


    model.train(

        data="datasets/merged/data.yaml",

        epochs=25,

        imgsz=640,

        batch=8,

        device=0,

        workers=0,

        trainer=YOLOCLTrainer,

        

        project="runs/joint_cl",

        name="yolo11s_supcon"

    )


if __name__ == "__main__":
    main()