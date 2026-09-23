from PIL import Image
from torch.utils.data import Dataset

from joint_cl.augmentations import get_contrastive_transform



class ContrastiveDataset(Dataset):

    def __init__(
        self,
        images,
        labels=None
    ):

        self.images = images
        self.labels = labels

        self.transform = get_contrastive_transform()


    def __len__(self):

        return len(self.images)


    def __getitem__(self,index):

        img_path = self.images[index]

        image = Image.open(
            img_path
        ).convert("RGB")


        view1 = self.transform(image)

        view2 = self.transform(image)


        if self.labels is not None:

            label = self.labels[index]

        else:

            label = -1


        return (
            view1,
            view2,
            label
        )