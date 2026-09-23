import torchvision.transforms as T


def get_contrastive_transform(size=640):

    return T.Compose([

        T.RandomResizedCrop(
            size=size,
            scale=(0.6,1.0)
        ),

        T.RandomHorizontalFlip(
            p=0.5
        ),

        T.RandomApply(
            [
                T.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1
                )
            ],
            p=0.8
        ),

        T.RandomGrayscale(
            p=0.2
        ),

        T.RandomApply(
            [
                T.GaussianBlur(
                    kernel_size=23
                )
            ],
            p=0.5
        ),

        T.ToTensor(),

        T.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )

    ])