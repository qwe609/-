# 1.create dataset
import os

from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, data_path: str, train=True, transform=None):
        self.data_path = data_path
        self.train_flag = train
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),  # convert PIL.Image to tensor, which is GY
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalization
                ]
            )
        else:
            self.transform = transform
        self.path_list = os.listdir(data_path)

    def __getitem__(self, idx: int):
        # img to tensor, label to tensor
        img_path = self.path_list[idx]
        abs_img_path = os.path.join(self.data_path, img_path)
        img = Image.open(abs_img_path)
        img = self.transform(img)

        if self.train_flag is True:
            if img_path.split('.')[0] == 'dog':
                label = 1
            else:
                label = 0
        else:
            label = int(img_path.split('.')[0])  # str-->int
        label = (torch.as_tensor(label,
                                 dtype=torch.int64))
        # must use long type, otherwise raise error when training, "expect long"
        return img, label

    def __len__(self) -> int:
        return len(self.path_list)

    # 2.dataset split
    def dataset_split(full_ds, train_rate):  # full_ds为train_ds, train_rate=0.8
        train_size = int(len(full_ds) * train_rate)
        validate_size = len(full_ds) - train_size
        train_dataset, validate_dataset = torch.utils.data.random_split(full_ds, [train_size, validate_size])
        return train_dataset, validate_dataset


def dataloader(dataset, batch_size):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=True, num_workers=4)
    return data_loader


# full_ds = MyDataset(data_path="./train", train=True)
# train_ds, validate_ds = MyDataset.dataset_split(full_ds, 0.8)
# for imgs, targets in train_ds:
#     print(imgs.shape, targets.shape)
#     print(imgs, targets)
#     break
#
# train_dataloader = torch.utils.data.DataLoader(dataset=train_ds,
#                                                batch_size=64)
# validate_dataloader = torch.utils.data.DataLoader(dataset=validate_ds,
#                                                   batch_size=64)
# writer = SummaryWriter('./logs')
# step = 0
# for imgs, targets in train_dataloader:
#     writer.add_images('imgs', imgs, step)
#     step += 1
#     break
# train_data_size = len(train_ds)
# test_data_size = len(validate_ds)
