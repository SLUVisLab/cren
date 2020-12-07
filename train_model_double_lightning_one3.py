from __future__ import absolute_import

from argparse import ArgumentParser
import os
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Callback
from pytorch_lightning import loggers as pl_loggers
from random import random
import torch
import time
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, models
from pytorch_lightning.callbacks import ModelCheckpoint

train_path = '/lab/vislab/DATA/Images/datasets_indoor/train/'
valid_path = '/lab/vislab/DATA/Images/datasets_indoor/val/'

num_classes = 67
class MetricCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


class Model(LightningModule):
    """ Model
    """

    def __init__(self, **kwargs):
        super(Model, self).__init__()

        self.epoch = 0
        self.learning_rate = 0.015
        self.training_correct_counter = 0
        self.training = False
        self.batch_size=4
        self.loss=nn.CrossEntropyLoss()

        mod1 = models.resnet50(pretrained=True)
        self.model1 = nn.Sequential(
            mod1.conv1,
            mod1.bn1,
            mod1.relu,
            mod1.maxpool,

            mod1.layer1,
            mod1.layer2,
            mod1.layer3,
            mod1.layer4,
        )
        self.fc = nn.Linear(2048, num_classes)
    def forward(self, x):

        bs, ncrops, c, h, w = x.size()
        #bs,c, h, w = x0.size()
        x = x.contiguous().view((-1, c, h, w))
        x = self.model1(x)
        # x0 = F.avg_pool2d(x0, 8)
        _, nf, h, w = x.size()

        x = x.view(bs, ncrops, nf, h, w).transpose(1, 2)
        xp=((F.softmax(x.reshape(bs,nf,-1),dim=-1)*64)>=((self.epoch-1)*1.0/25)).reshape(bs,nf,ncrops,h,w)
        x_rate = torch.sum(xp) * 1.0 / (bs * ncrops * nf * h * w)
        x=(torch.sum(x*xp,[3,4])/(torch.sum(xp,[3,4])+0.001)).transpose(1,2)
        x = x.view(bs, ncrops, -1)
        x = torch.mean(x, 1)
        #x=torch.stack([torch.where(ax>=torch.mean(ax,0)+1, ax, torch.tensor(float("nan")).to("cuda:0")) for ax in x])
        #x=torch.stack([torch.sum(torch.where(ax==ax,ax,torch.tensor(0.).to("cuda:0")),0)/(torch.sum(~torch.isnan(ax),0)+1) for ax in x])
        if self.training == True:
            x = F.dropout(x, 0.2)
        return self.fc(x.view(x.size(0), -1)), x_rate
    def train(self):
        self.model1.train()
        self.fc.train()
    def eval(self):
        self.model1.eval()
        self.fc.eval()
    def prepare_data(self):

        data_transforms = {
            'train': transforms.Compose([

                transforms.Lambda(lambda img: self.RandomErase(img)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Lambda(lambda img: self.crops_and_random(img)),
                transforms.Lambda(lambda crops: torch.stack([transforms.Compose([transforms.ToTensor(),
                                                                                 transforms.Normalize(mean=[n / 255.
                                                                                                            for n in
                                                                                                            [75.58, 96.37, 92.88]],
                                                                                                      std=[n / 255. for
                                                                                                           n in
                                                                                                           [43.36, 53.14, 52.06]])])(
                    crop) for
                                                             crop in crops]))

            ]),

            # currently same as train
            'valid': transforms.Compose([
                transforms.Lambda(lambda img: self.val_crops(img)),
                transforms.Lambda(lambda crops: torch.stack([transforms.Compose([transforms.ToTensor(),
                                                                                 transforms.Normalize(mean=[n / 255.
                                                                                                            for n in
                                                                                                            [75.58, 96.37, 92.88]],
                                                                                                      std=[n / 255. for
                                                                                                           n in
                                                                                                           [43.36, 53.14, 52.06]])])(
                    crop) for
                                                             crop in crops]))

            ]),
        }

        self.trainset = datasets.ImageFolder(train_path, data_transforms['train'])
        self.validset = datasets.ImageFolder(valid_path, data_transforms['valid'])

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, sampler=None, num_workers=32)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, shuffle=False, sampler=None, num_workers=32)

    def configure_optimizers(self):
        optimizer=torch.optim.SGD(self.parameters(), lr=self.learning_rate,momentum=0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        self.train()
        # for imgs, labels in model_ft.trainset:
        #     print(labels)

        self.training = True
        inputs, labels = batch
        outputs, x_rate = self(inputs)
        loss = self.loss(outputs, labels)

        labels_hat = torch.argmax(outputs, dim=1)
        train_acc = torch.sum(labels.data == labels_hat).item() / (len(labels) * 1.0)

        self.eval()

        return {
            "tra_rate": x_rate,
            'loss': loss,
            'train_acc': train_acc
        }

    def training_epoch_end(self, training_step_outputs):
        self.training = True
        self.train()

        train_acc = np.mean([x['train_acc'] for x in training_step_outputs])
        train_acc = torch.tensor(train_acc, dtype=torch.float32)
        print("train_acc", train_acc)
        train_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        x_rate = torch.stack([x['tra_rate'] for x in training_step_outputs]).mean()
        # self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.epoch)
        self.eval()

        return {
            'log': {
                'train_loss': train_loss,
                'train_acc': train_acc,
                "tra_rate": x_rate,
            },
            'progress_bar': {
                'train_loss': train_loss,
                'train_acc': train_acc,
                "tra_rate": x_rate
            }
        }

    def validation_step(self, batch, batch_idx):
        self.training = False
        self.eval()

        inputs, labels = batch
        outputs, x_rate= self(inputs)
        loss = self.loss(outputs, labels)

        # _, preds = torch.max(outputs, 1)
        # running_corrects += torch.sum(preds == labels.data)

        labels_hat = torch.argmax(outputs, dim=1)
        # print("labels", labels,"labels_hat",labels_hat)
        val_acc = torch.sum(labels.data == labels_hat).item() / (len(labels) * 1.0)

        self.train()
        return {
            "val_rate": x_rate,
            'val_loss': loss,
            'val_acc': val_acc
        }

    def validation_epoch_end(self, validation_step_outputs):
        self.training = False
        self.eval()

        val_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()
        # val_tot = [x['val_acc'] for x in validation_step_outputs]
        # val_acc = np.mean(val_tot)
        print("HERE\n\n\n\nValidation in each step\n")
        print([x['val_acc'] for x in validation_step_outputs])
        x_rate = torch.stack([x['val_rate'] for x in validation_step_outputs]).mean()
        val_acc = np.mean([x['val_acc'] for x in validation_step_outputs])
        val_acc = torch.tensor(val_acc, dtype=torch.float32)
        print("val_loss", val_loss)
        print("val_acc", val_acc)

        self.epoch += 1
        self.train()
        return {
            'log': {
                'val_loss': val_loss,
                'val_acc': val_acc,
                "val_rate": x_rate,
            },
            'progress_bar': {
                'val_loss': val_loss,
                'val_acc': val_acc,
                "val_rate": x_rate,
            }
        }

    def random_crops(self, img, k, s):
        crops = []
        rand = torchvision.transforms.RandomCrop(s)
        Res=torchvision.transforms.Resize(512,interpolation=2)
        for j in range(k):
            im = Res(rand(img))
            crops.append(im)
        return crops

    def crops_and_random(self, img):

        res = torchvision.transforms.Resize(1024, interpolation=2)
        img=res(img)
        #Rand = torchvision.transforms.RandomCrop(512)
        #img1 = Rand(img)
        #Res = torchvision.transforms.Resize(256, interpolation=2)
        #crop512=self.random_crops(img, 4, s=512)
        #crop64=[]
        #for c in crop512:
            #crop64.append(self.random_crops(c, 1, s=64)[0])
        return self.random_crops(img,4, 512)

    def val_crops(self, img):
        res = torchvision.transforms.Resize(1024, interpolation=2)
        img = res(img)
        Cent1024= torchvision.transforms.CenterCrop((1024,1024))
        Cent256 = torchvision.transforms.CenterCrop((1024,1024))
        img1=np.array(Cent1024(img))
        #im = np.array(img)
        re = torchvision.transforms.Resize((512, 512), interpolation=2)
        im=np.array(Cent256(img))

        crs512 = []
        for i in range(2):
            for j in range(2):
                crs512.append(
                    re(Image.fromarray((img1[i * 512:((i + 1) * 512), j *512:((j + 1) * 512)]).astype('uint8')).convert(
                        'RGB')))

        return crs512

    def RandomErase(self, img, p=0.5, s=(0.06, 0.12), r=(0.5, 1.5)):
        im = np.array(img)
        w, h, _ = im.shape
        S = w * h
        pi = random()
        if pi > p:
            return img
        else:
            Se = S * (random() * (s[1] - s[0]) + s[0])
            re = random() * (r[1] - r[0]) + r[0]
            He = int(np.sqrt(Se * re))
            We = int(np.sqrt(Se / re))
            if He >= h:
                He = h - 1
            if We >= w:
                We = w - 1
            xe = int(random() * (w - We))
            ye = int(random() * (h - He))
            im[xe:xe + We, ye:ye + He] = int(random() * 255)
            return Image.fromarray(im.astype('uint8')).convert('RGB')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.1)
        return parser


if __name__ == '__main__':
    metrics_callback = MetricCallback()
    log_dir="lightning_one_3"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logger = pl_loggers.TensorBoardLogger(log_dir)
    checkpoint_callback = ModelCheckpoint(
        period=5,
        monitor='val_acc',
        filepath=log_dir+'/sample-mit-{epoch:02d}-{val_acc:.2f}',
        save_top_k = 1,
        mode = 'max')
    trainer = pl.Trainer(
        check_val_every_n_epoch=5,
        max_epochs=25,
        gpus=[3] if torch.cuda.is_available() else None,
        callbacks=[metrics_callback],
        logger=logger
    )

    model_ft = Model()
    ct = 0
    for child in model_ft.model1.children():
        ct += 1
        if ct < 5:  # freezing the first few layers to prevent overfitting
            for param in child.parameters():
                param.requires_grad = False

    trainer.fit(model_ft)