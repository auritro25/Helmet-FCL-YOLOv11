from ultralytics import YOLO


model = YOLO(
    "models/phase2_supcon_best.pt"
)


model.val(

    data="datasets/merged/data.yaml",

    imgsz=640,

    batch=8,

    device=0,

    workers=0

)