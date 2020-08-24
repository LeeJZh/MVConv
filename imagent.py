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
import torch.nn.functional as F
from torch.nn.modules import padding
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule


class MVConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 bias) -> None:
        super().__init__()
        projected_channels = max(8, in_channels // 3 + 1)

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

    def forward(self, x):
        return self.main_conv(x) + self.left_conv(
            self.left_projection(x)) + self.right_conv(
                self.right_projection(x))

    @staticmethod
    def merge_side(curr: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        with torch.no_grad:
            curr_t = curr.permute([2, 3, 0, 1])
            prev_t = prev.permute([2, 3, 0, 1])
            merged = torch.matmul(curr_t, prev_t)
            return merged.permute([2, 3, 0, 1])

    def deploy(self):
        with torch.no_grad():
            deploy_conv = torch.nn.Conv2d(self.in_channels,
                                          self.out_channels,
                                          self.kernel_size,
                                          self.stride,
                                          self.padding,
                                          bias=self.bias)
            deploy_conv.weight = self.main_conv.weight
            deploy_conv.weight += self.merge_side(self.left_conv.weight,
                                                  self.left_projection.weight)
            deploy_conv.weight += self.merge_side(self.right_conv.weight,
                                                  self.right_projection.weight)
            if self.bias is True:
                deploy_conv.bias = self.main_conv.bias + self.left_conv.bias + self.right_conv.bias
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


def inplace_module_modification(m: torch.nn.Module):
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
                       bias=child.bias is not None))
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
        model = models.__dict__[self.arch](pretrained=self.pretrained)
        self.block = block
        if self.block != "base":
            inplace_module_modification(model)
        self.model = model
        self.example_input_array = torch.randn((1, 3, 224, 224))
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
        loss_val = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))

        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'acc1': acc1,
            'acc5': acc5,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_val = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc1': acc1,
            'val_acc5': acc5,
        })
        return output

    def validation_epoch_end(self, outputs):
        tqdm_dict = {}
        for metric_name in ["val_loss", "val_acc1", "val_acc5"]:
            tqdm_dict[metric_name] = torch.stack(
                [output[metric_name] for output in outputs]).mean()

        result = {
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
            'val_loss': tqdm_dict["val_loss"]
        }
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
        # scheduler = lr_scheduler.LambdaLR(optimizer,
        #                                   lambda epoch: 0.1**(epoch // 30))
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
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        train_dir = os.path.join(self.data_path, 'train')
        print("building training set")
        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
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
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        val_dir = os.path.join(self.data_path, 'val')
        print("building validation dataset")
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                val_dir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
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

    def test_epoch_end(self, *args, **kwargs):
        outputs = self.validation_epoch_end(*args, **kwargs)

        def substitute_val_keys(out):
            return {k.replace('val', 'test'): v for k, v in out.items()}

        outputs = {
            'test_loss': outputs['val_loss'],
            'progress_bar': substitute_val_keys(outputs['progress_bar']),
            'log': substitute_val_keys(outputs['log']),
        }
        return outputs

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument(
            '-a',
            '--arch',
            metavar='ARCH',
            default='resnet18',
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
    print(args)
    model = ImageNetLightningModel(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args)

    if args.evaluate:
        trainer.test(model)
    else:
        trainer.fit(model)


def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('--data-path',
                               default="/workspace/cpfs-data/ImageNet",
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
    parser = ImageNetLightningModel.add_model_specific_args(parent_parser)
    parser.set_defaults(
        profiler=True,
        deterministic=True,
        max_epochs=90,
    )
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()