"""
Author: Leandro Lima
"""

import torch
from torch import nn
import torchvision
import pytorch_lightning as pl

# Only works when loaded twice
try:
    from lightly.data import LightlyDataset
except:
    from lightly.data import LightlyDataset

from lightly.data import SimCLRCollateFunction
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
import timm


class SimCLR(pl.LightningModule):
    def __init__(self, backbone_model, dataset, optimizer="SGD", optimizer_args=None, num_workers=8, batch_size=30,
                 head_size=2048, input_size=224):
        """
            SimCLR model
        :param backbone_model: Backbone model
        :param dataset: Pytorch dataset
        :param optimizer: Optimizer
        :param num_workers: Number of workers for the dataloader
        :param batch_size: Batch size for the dataloader
        :param head_size: Head size for the projection head
        :param input_size: Input image size of the backbone model
        """
        super().__init__()

        # Create dataloader
        ds = LightlyDataset.from_torch_dataset(dataset)
        collate_fn = SimCLRCollateFunction(
            input_size=input_size,
            gaussian_blur=0.,
        )
        self.dataloader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        )

        self.optimizer = optimizer

        if optimizer_args is None:
            optimizer_args = {"lr": 0.06}
        self.optimizer_args = optimizer_args

        self.backbone = backbone_model
        self.projection_head = SimCLRProjectionHead(backbone_model.num_features, head_size, head_size)
        self.criterion = NTXentLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        return loss

    def configure_optimizers(self):
        if self.optimizer == "SGD":
            optim = torch.optim.SGD(self.parameters(), **self.optimizer_args)
        else:
            raise ValueError("Optimizer not implemented")
        return optim

    def get_dataloader(self):
        return self.dataloader


class SSLModel:
    def __init__(self, model_name, ssl_name, dataset, max_epochs=10, batch_size=30, num_workers=8, optimizer="SGD",
                 optimizer_args=None, head_size=2048):
        """
            Self-supervised learning
        :param model_name: A TIMM model name. Eg. 'mobilenetv2_100'
        :param dataset: A pytorch dataset
        :param max_epochs: Maximum number of epochs to train for
        :param batch_size: Batch size for the dataloader
        :param num_workers: Number of workers for the dataloader
        :param optimizer: Optimizer name. Currently only SGD is supported.
        :param optimizer_args: Optimizer arguments dictionary. Eg. {"lr": 0.06}
        :param head_size: Head size for the projection head.
        """
        self.dataset = dataset
        # self.dataset = torchvision.datasets.FakeData(size=100)
        self.model_name = model_name
        self.ssl_name = ssl_name
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.head_size = head_size

        self.model = self._create_backbone_model(self.model_name)
        self.SSLmodel = self._create_ssl_model(ssl_name)
        self.trainer = pl.Trainer(max_epochs=self.max_epochs, accelerator="gpu", devices=-1, auto_select_gpus=True)

    def train(self):
        self.trainer.fit(model=self.SSLmodel, train_dataloaders=self.SSLmodel.get_dataloader())

    def get_model(self):
        return self.model

    def _create_backbone_model(self, model_name):
        timm_model = timm.create_model(model_name, pretrained=True, num_classes=0)
        return timm_model

    def _create_ssl_model(self, ssl_name):
        if ssl_name == "simclr":
            return SimCLR(
                self.model,
                self.dataset,
                input_size=self._get_input_size(self.model),
                optimizer=self.optimizer,
                optimizer_args=self.optimizer_args,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                head_size=self.head_size
            )
        else:
            raise Exception("Unsupported SSL model")

    def _get_input_size(self, model):
        size = list(model.default_cfg["input_size"][1:])[0]
        return size


if __name__ == "__main__":
    model = SSLModel("mobilenetv2_100", "simclr", None)
    model.train()
