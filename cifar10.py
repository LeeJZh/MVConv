"""
This example is largely adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
Before you can run this example, you will need to download the ImageNet dataset manually from the
`official website <http://image-net.org/download>`_ and place it into a folder `path/to/imagenet`.
Train on ImageNet with default parameters:
.. code-block: bash
    python imagenet.py --data_root /path/to/imagenet
or show all options you can change:
.. code-block: bash
    python imagenet.py --help
"""
import os
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import parameter
import torch.nn.functional as F
from torch.nn.modules import padding
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils import data
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule

import cifar_models

# PROJECTION_RATIO = 3


class MVConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 bias, PROJECTION_RATIO) -> None:
        super().__init__()
        projected_channels = max(8, in_channels // PROJECTION_RATIO)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.left_projection = torch.nn.Conv2d(in_channels,
                                               projected_channels,
                                               kernel_size=1,
                                               stride=1,
                                               padding=0,
                                               bias=False)
        self.right_projection = torch.nn.Conv2d(in_channels,
                                                projected_channels,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                bias=False)
        self.main_conv = torch.nn.Conv2d(in_channels,
                                         out_channels,
                                         kernel_size,
                                         stride,
                                         padding,
                                         bias=bias)
        self.left_conv = torch.nn.Conv2d(projected_channels,
                                         out_channels,
                                         kernel_size,
                                         stride,
                                         padding,
                                         bias=bias)
        self.right_conv = torch.nn.Conv2d(projected_channels,
                                          out_channels,
                                          kernel_size,
                                          stride,
                                          padding,
                                          bias=bias)
        self.main_bn = nn.BatchNorm2d(out_channels)
        self.left_bn = nn.BatchNorm2d(out_channels)
        self.right_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.main_bn(self.main_conv(x)) + self.left_bn(
            self.left_conv(self.left_projection(x))) + self.right_bn(
                self.right_conv(self.right_projection(x)))

    @staticmethod
    def merge_side(curr: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        with torch.no_grad:
            curr_t = curr.permute([2, 3, 0, 1])
            prev_t = prev.permute([2, 3, 0, 1])
            merged = torch.matmul(curr_t, prev_t)
            return merged.permute([2, 3, 0, 1])

    @staticmethod
    def fuse_kernel(conv: nn.Conv2d, bn: nn.BatchNorm2d):
        with torch.no_grad():
            conv_weight = conv.weight
            conv_bias = conv.bias
            bn_gamma = bn.weight
            bn_beta = bn.bias
            bn_std = bn.running_var
            bn_mean = bn.running_mean
            bn_eps = bn.eps
            if conv_bias is not None:
                bn_mean = bn_mean - conv_bias
                bn_mean = bn_mean.view(-1, 1, 1, 1)
            else:
                bn_mean = bn_mean.view(-1, 1, 1, 1)
            if bn_std is not None:
                bn_std = bn_std + bn_eps
            else:
                bn_std = bn_eps
            if bn_gamma is not None:
                bn_gamma = bn_gamma.view(-1, 1, 1, 1)
            else:
                bn_gamma = torch.ones_like(bn_std)
            if bn_beta is not None:
                bn_beta = bn_beta.view(-1, 1, 1, 1)
            else:
                bn_beta = torch.zeros_like(bn_mean)
            weight = conv_weight * bn_gamma / bn_std
            bias = bn_beta - bn_gamma * bn_mean / bn_std
            bias = bias.view(-1)

        return weight, bias

    def deploy(self):
        with torch.no_grad():
            deploy_conv = torch.nn.Conv2d(self.in_channels,
                                          self.out_channels,
                                          self.kernel_size,
                                          self.stride,
                                          self.padding,
                                          bias=True)
            fused_main_weight, fused_main_bias = self.fuse_kernel(
                self.main_conv, self.main_bn),
            fused_left_weight, fused_left_bias = self.fuse_kernel(
                self.left_conv, self.left_bn),
            fused_right_weght, fused_right_bias = self.fuse_kernel(
                self.right_conv, self.right_bn)

            deploy_conv.weight = fused_main_weight + fused_left_weight + fused_right_weght
            deploy_conv.bias = fused_main_bias + fused_left_bias + fused_right_bias

        return deploy_conv


def make_it_deploy(m: torch.nn.Module):
    for name, child in m.named_children():
        if isinstance(child, MVConv):
            # print("make {} it deploy".format(name))
            # print(child)
            setattr(m, name, child.deploy())
            # print(getattr(m, name))
        else:
            inplace_module_modification(child)


def inplace_module_modification(m: torch.nn.Module, project_ratio: float):
    for name, child in m.named_children():
        if isinstance(child, torch.nn.Conv2d) and child.kernel_size[0] == 3:
            # print("replace {} with mvconv".format(name))
            # print(child)
            setattr(
                m, name,
                MVConv(child.in_channels,
                       child.out_channels,
                       child.kernel_size,
                       child.stride,
                       child.padding,
                       bias=child.bias is not None,
                       PROJECTION_RATIO=project_ratio))
            setattr(m, name.replace('conv', 'bn'), nn.Identity())
            # print(getattr(m, name))
        else:
            inplace_module_modification(child)


class ImageNetLightningModel(LightningModule):

    # pull out resnet names from torchvision models
    MODEL_NAMES = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    def __init__(
        self,
        arch: str,
        pretrained: bool,
        lr: float,
        momentum: float,
        weight_decay: int,
        data_path: str,
        batch_size: int,
        workers: int,
        block: str,
        project_ratio: float,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.arch = arch
        self.pretrained = pretrained
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.data_path = data_path
        self.batch_size = batch_size
        self.workers = workers
        # model = models.__dict__[self.arch](pretrained=self.pretrained)
        model = getattr(cifar_models, arch.lower())(num_classes=10)
        self.block = block
        self.project_ratio = project_ratio
        if self.block != "base":
            inplace_module_modification(model,
                                        project_ratio=self.project_ratio)
        self.model = model
        self.example_input_array = torch.zeros((1, 3, 32, 32))
        print(self.hparams)

        total_devices = self.hparams.gpus * self.hparams.num_nodes
        train_batches = len(self.train_dataloader()) // total_devices
        self.train_steps = (self.hparams.max_epochs * train_batches
                            ) // self.hparams.accumulate_grad_batches

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))

        result = pl.TrainResult(minimize=loss)
        result.log('train_loss',
                   loss,
                   on_step=True,
                   on_epoch=True,
                   prog_bar=True,
                   logger=True,
                   sync_dist=True)
        result.log('train_acc1',
                   acc1,
                   on_step=True,
                   on_epoch=True,
                   prog_bar=True,
                   logger=True,
                   sync_dist=True)
        result.log('train_acc5',
                   acc5,
                   on_step=True,
                   on_epoch=True,
                   prog_bar=False,
                   logger=True,
                   sync_dist=True)

        return result

    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))

        result = pl.EvalResult(checkpoint_on=acc1, early_stop_on=acc1)
        result.log('val_loss',
                   loss,
                   on_step=False,
                   on_epoch=True,
                   prog_bar=True,
                   logger=True,
                   sync_dist=True)
        result.log('val_acc1',
                   acc1,
                   on_step=False,
                   on_epoch=True,
                   prog_bar=True,
                   logger=True,
                   sync_dist=True)
        result.log('val_acc5',
                   acc5,
                   on_step=False,
                   on_epoch=True,
                   prog_bar=False,
                   logger=True,
                   sync_dist=True)

        return result

    @staticmethod
    def __accuracy(output, target, topk=(1, )):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(),
                              lr=self.lr,
                              momentum=self.momentum,
                              weight_decay=self.weight_decay)
        scheduler = lr_scheduler.LambdaLR(optimizer,
                                          lambda epoch: 0.1**(epoch // 30))
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=self.train_steps)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',  # or 'epoch'
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        )

        print("building training set")
        train_dataset = datasets.CIFAR10(self.data_path,
                                         train=True,
                                         download=True,
                                         transform=transforms.Compose([
                                             transforms.RandomCrop(
                                                 32,
                                                 padding=4,
                                                 padding_mode='reflect'),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(), normalize
                                         ]))
        print("training set done")
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
        )
        return train_loader

    def val_dataloader(self):
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        )
        print("building validation dataset")
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(self.data_path,
                             train=False,
                             download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )
        print("validation set done")
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument(
            '-a',
            '--arch',
            metavar='ARCH',
            default='resnet56',
            choices=ImageNetLightningModel.MODEL_NAMES,
            help=('model architecture: ' +
                  ' | '.join(ImageNetLightningModel.MODEL_NAMES) +
                  ' (default: resnet18)'))
        parser.add_argument('-j',
                            '--workers',
                            default=4,
                            type=int,
                            metavar='N',
                            help='number of data loading workers (default: 4)')
        parser.add_argument(
            '-b',
            '--batch-size',
            default=256,
            type=int,
            metavar='N',
            help='mini-batch size (default: 256), this is the total '
            'batch size of all GPUs on the current node when '
            'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr',
                            '--learning-rate',
                            default=0.1,
                            type=float,
                            metavar='LR',
                            help='initial learning rate',
                            dest='lr')
        parser.add_argument('--momentum',
                            default=0.9,
                            type=float,
                            metavar='M',
                            help='momentum')
        parser.add_argument('--wd',
                            '--weight-decay',
                            default=1e-4,
                            type=float,
                            metavar='W',
                            help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        parser.add_argument('--pretrained',
                            dest='pretrained',
                            action='store_true',
                            help='use pre-trained model')
        parser.add_argument('--block',
                            dest='block',
                            default='base',
                            type=str,
                            metavar='BK')
        parser.add_argument('--project-ratio',
                            default=3.0,
                            type=float,
                            dest='project_ratio')
        return parser


def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)

    if args.distributed_backend == 'ddp':
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / max(1, args.gpus))
        args.workers = int(args.workers / max(1, args.gpus))
    # print(**vars(args))
    # print(args)
    model = ImageNetLightningModel(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args)

    if args.evaluate:
        try:
            model = model.load_from_checkpoint(args.ckpt)
        except Exception as excep:
            print(excep)
        make_it_deploy(model)
        trainer.test(model)
    else:
        trainer.fit(model)


def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('--data-path',
                               default="/workspace/cpfs-data/cifar10",
                               metavar='DIR',
                               type=str,
                               help='path to dataset')
    parent_parser.add_argument('-e',
                               '--evaluate',
                               dest='evaluate',
                               action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('--seed',
                               type=int,
                               default=42,
                               help='seed for initializing training.')
    parent_parser.add_argument('--ckpt', type=str, default='')
    parser = ImageNetLightningModel.add_model_specific_args(parent_parser)
    parser.set_defaults(
        profiler=True,
        deterministic=True,
        max_epochs=400,
    )
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()
