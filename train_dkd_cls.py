import os.path
from functools import cached_property
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from torch.utils.data import Dataset

from model import Model
from trainer import Trainer
from util.logger import get_logger

torch.multiprocessing.set_sharing_strategy('file_system')

logger = get_logger(__name__)

import albumentations as aug
import albumentations.pytorch as aug_torch
import cv2


def test_transform(image_size=512):
    return aug.Compose([
        aug.SmallestMaxSize(max_size=image_size, always_apply=True),
        aug.CenterCrop(image_size, image_size, always_apply=True),
        aug.ToFloat(always_apply=True),
        aug_torch.ToTensorV2(),
    ])


def train_transform(image_size=512):
    return aug.Compose([
        aug.SmallestMaxSize(max_size=image_size, always_apply=True),
        aug.CenterCrop(image_size, image_size, always_apply=True),
        aug.Flip(p=0.2),
        aug.ImageCompression(quality_lower=10, quality_upper=80, p=0.2),
        aug.MedianBlur(p=0.3),
        aug.RandomBrightnessContrast(p=0.3),
        aug.RandomGamma(p=0.2),
        aug.GaussNoise(p=0.3),
        aug.Rotate(border_mode=cv2.BORDER_CONSTANT, value=0, p=0.2, limit=45),
        aug.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.3),
        aug.Cutout(num_holes=8, max_h_size=16, max_w_size=16, fill_value=0, always_apply=False, p=0.2),
        aug.ToFloat(always_apply=True),
        aug_torch.ToTensorV2(),
    ])


class DKD_dataset(Dataset):
    def __init__(self, split='train', transform=None):
        super().__init__()
        root = './data'
        if split == 'train':
            self.data = pandas.read_csv(os.path.join(root, 'dkd-train.csv'))
        elif split == 'test':
            self.data = pandas.read_csv(os.path.join(root, 'dkd-test.csv'))
        else:
            raise ValueError('split must be train or test')
        self.image_root = './image/DKD'
        self.image_column = ['image']
        self.label_column = ['DKD']
        self.transform = transform
        self.images = []
        self.labels = []
        self.get_data()
        self.data_len = len(self.labels)

    @cached_property
    def pos_weight(self) -> torch.FloatTensor:
        w = [(self.data['DKD'] > i).mean() for i in range(3)]
        return torch.FloatTensor(w)

    def get_data(self):
        data = self.data
        for idx, obj in data.iterrows():
            img_list = str(obj[self.image_column[0]]).split(',')

            for img in img_list[:]:
                image_path = os.path.join(self.image_root, img)
                if os.path.exists(image_path):
                    self.images.append(img)
                    self.labels.append(torch.tensor(obj[self.label_column[0]], dtype=torch.long))
                else:
                    continue

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        return self._item(item)

    def _item(self, item):
        try:
            image_path = os.path.join(self.image_root, self.images[item])
            if not os.path.exists(image_path):
                raise FileNotFoundError
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        except Exception as e:
            print(e)
        if self.transform is not None:
            image = self.transform(image=image)['image']
        return image, self.labels[item]

class DKDCls(Trainer):
    @cached_property
    def pos_weight(self):
        dd = DKD_dataset()
        w = dd.pos_weight
        w = w.to(self.device)
        return w

    @cached_property
    def model(self):
        model = Model(
            backbone=self.cfg.model,
            output_size=4
        )
        return model

    @cached_property
    def train_dataset(self) -> Dataset:
        return DKD_dataset(split='train', transform=train_transform(self.cfg.image_size))

    @cached_property
    def test_dataset(self) -> Dataset:
        return DKD_dataset(split='test', transform=test_transform(self.cfg.image_size))

    def batch(self, epoch, i_batch, data) -> dict:
        img, label = data
        img = img.to(self.device)
        target_bin = [label > i for i in range(3)]
        target_bin = torch.stack(target_bin, dim=1)
        target_bin = target_bin.to(torch.float)
        target_bin = target_bin.to(self.device)
        target_level = label.to(self.device)
        pred = self.model(img)

        # pred_level = self.binary_to_level(pred)
        pred_level = pred
        pred_bin = self.level_to_binary(torch.softmax(pred_level, dim=1))

        loss = F.binary_cross_entropy_with_logits(
            pred_bin, target_bin, pos_weight=self.pos_weight) * 0.3

        loss2 = F.cross_entropy(pred_level, target_level) * 0.7
        loss += loss2
        return dict(loss=loss, pred_level=pred_level, label=target_level, pred_bin=pred_bin)

    def binary_to_level(self, pred: torch.Tensor):
        result = torch.ones((pred.size(0), 4), device=pred.device)
        pred = torch.sigmoid(pred)
        for i in range(4):
            # label i
            for j in range(3):
                # j target
                if j < i:
                    result[:, i] *= pred[:, j]
                else:
                    result[:, i] *= 1 - pred[:, j]
        return result

    def level_to_binary(self, pred: torch.Tensor):
        result = torch.ones((pred.size(0), 3), device=pred.device)
        for i in range(3):
            oo = torch.sum(pred[:, i+1:], dim=1, keepdim=False)
            # print(oo.shape)
            result[:, i] += oo
        return result

    def matrix(self, epoch, data) -> dict:
        pred_level = data['pred_level']
        pred_bin = data['pred_bin']
        label = data['label']
        # one_hot_dist = self.binary_to_level(pred)
        pred_level = torch.argmax(pred_level, dim=1)
        mat = {}
        mat['mean_loss'] = torch.mean(data['loss']).item()
        for i in range(3):
            try:
                mat[f'auc_{i}'] = roc_auc_score(label > i, pred_bin[:, i])
            except Exception as e:
                print(e)
        mat['kappa'] = float(cohen_kappa_score(label, pred_level, weights='quadratic'))
        mat['acc'] = torch.mean((pred_level == label).to(torch.float)).item()

        return mat


if __name__ == '__main__':
    import os
    os.environ['model'] = 'resnet50'
    os.environ['lr'] = '0.00001'
    os.environ['batch_size'] = '48'
    os.environ['image_size'] = '512'
    os.environ['epochs'] = '100'
    os.environ['device'] = 'cuda:0'
    os.environ['num_workers'] = '8'
    os.environ['load_pretrain'] = "model/pretrain_678.pth"
    trainer = DKDCls()
    trainer.train()
