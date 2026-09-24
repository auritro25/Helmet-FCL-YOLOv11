from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


INPUT = Path(
    "thesis_results/detection_examples"
)


OUTPUT = Path(
    "thesis_results/graphs"
)


OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


MODELS = [
    "centralized",
    "fedavg",
    "fedprox"
]


images = sorted(
    list(
        (INPUT / "centralized").glob("*.jpg")
    )
)


# choose 5 examples for thesis figure

images = images[:5]


thumb_width = 400
thumb_height = 300


title_height = 50


canvas_width = (
    thumb_width * 3
)

canvas_height = (
    (thumb_height + title_height)
    *
    len(images)
)


canvas = Image.new(
    "RGB",
    (
        canvas_width,
        canvas_height
    ),
    "white"
)


try:
    font = ImageFont.truetype(
        "arial.ttf",
        30
    )

except:

    font = None



for row,img_name in enumerate(images):

    for col,model in enumerate(MODELS):

        img_path = (
            INPUT /
            model /
            img_name.name
        )


        img = Image.open(
            img_path
        ).convert(
            "RGB"
        )


        img.thumbnail(
            (
                thumb_width,
                thumb_height
            )
        )


        x = col * thumb_width

        y = row * (
            thumb_height +
            title_height
        )


        canvas.paste(
            img,
            (
                x,
                y + title_height
            )
        )


        draw = ImageDraw.Draw(
            canvas
        )


        draw.text(
            (
                x + 10,
                y + 10
            ),
            model.upper(),
            fill="black",
            font=font
        )



output = (
    OUTPUT /
    "detection_comparison_grid.png"
)


canvas.save(
    output,
    dpi=(300,300)
)


print(
    "Saved:",
    output
)