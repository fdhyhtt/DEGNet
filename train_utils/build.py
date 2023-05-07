from backbone import resnet50_fpn_backbone, r50_ffm
import bisect
import torch
import copy
from . import samplers
from .samplers import GroupedBatchSampler, create_aspect_ratio_groups
import os
import importlib
import dataset as D
from .lr_schduler import WarmupMultiStepLR


def import_file(module_name, file_path=None):
    module = importlib.machinery.SourceFileLoader(module_name, file_path).load_module()
    return module


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler

def make_swin_optimizer(args, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = 0.0001
        weight_decay = 0.05
        if 'density_head' in key:
            lr = lr * 1e-5 * 1
            weight_decay = 5e-4
        elif "relative_position_bias_table" in key:
            weight_decay = 0
        elif "norm" in key:
            weight_decay = 0
        elif "bias" in key:
            lr = args.lr * 2
            weight_decay = 0

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    # optimizer = torch.optim.SGD(params, lr, momentum=args.momentum, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(params, lr=0.0001, betas=(0.9, 0.999), weight_decay=5E-2)

    return optimizer

def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    if aspect_grouping >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=aspect_grouping)
        # 每个batch图片从同一高宽比例区间中取
        batch_sampler = GroupedBatchSampler(
            sampler, group_ids, images_per_batch
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def building_optimizer(cfg, model):
    params = []
    base_lr = cfg['opt_param']['lr']
    base_weight_decay = cfg['opt_param']['weight_decay']
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = base_lr
        weight_decay = base_weight_decay

        if "bias" in key:
            lr = base_lr * 2
            weight_decay = 0

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg['opt_param']['momentum'], weight_decay=base_weight_decay)

    return optimizer


def make_lr_scheduler(cfg, optimizer):
    lr_steps = cfg['Lr_param']['lr_steps']
    lr_gamma = cfg['Lr_param']['lr_gamma']
    if cfg['Lr_param']['use_epoch']:
        return torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=lr_steps,
                                                    gamma=lr_gamma)
    return WarmupMultiStepLR(
        optimizer,
        lr_steps,
        lr_gamma,
        warmup_factor=1.0 / 1000,
        warmup_iters=1000,
        warmup_method='linear',
    )



def build_dataset(dataset_list, transforms, dataset_catalog, is_train=True):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_trian, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)
        factory = getattr(D, data["factory"])
        args = data["args"]
        # during training
        args["transforms"] = transforms
        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def make_dataloader(cfg, dataset, is_train=True, nw=0):
    batch_size = cfg['Dataset']['batch_size']
    if is_train:
        for dt in dataset:
            if cfg['Dataset']['aspect_ratio_factor'] >= 0:
                train_sampler = torch.utils.data.RandomSampler(dt)
                # Count the position index of image aspect ratios in the bins interval
                group_ids = create_aspect_ratio_groups(dataset, k=cfg['Dataset']['aspect_ratio_factor'])
                train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)
                data_loader = torch.utils.data.DataLoader(dt,
                                                          batch_sampler=train_batch_sampler,
                                                          # pin_memory=True,
                                                          num_workers=nw,
                                                          collate_fn=dt.collate_fn)
            else:
                data_loader = torch.utils.data.DataLoader(dt,
                                                          batch_size=batch_size,
                                                          shuffle=True,
                                                          # pin_memory=True,
                                                          num_workers=nw,
                                                          collate_fn=dt.collate_fn)


    else:
        data_loader = torch.utils.data.DataLoader(dataset[0],
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=dataset[0].collate_fn)
    return data_loader


def make_dataset(dataset_list, transforms, is_train=True):
    PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
    # PATHS_CATALOG = 'train_utils.paths_catalog'
    # 拿函数的类
    paths_catalog = import_file(PATHS_CATALOG, PATHS_CATALOG)
    DatasetCatalog = paths_catalog.DatasetCatalog
    transforms = transforms
    datasets = build_dataset(dataset_list, transforms, DatasetCatalog, is_train=is_train)

    return datasets


def make_iter_data_loader(cfg, is_distributed=False, start_iter=0, datasets=None, nw=0):
    num_gpus = 1
    images_per_batch = cfg['Dataset']['batch_size']  # 8
    if cfg['Dataset'].get('train_s') and cfg['Dataset'].get('train_t'):
        images_per_batch = images_per_batch//2
    assert (
        images_per_batch % num_gpus == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
    "of GPUs ({}) used.".format(images_per_batch, num_gpus)
    images_per_gpu = images_per_batch // num_gpus
    shuffle = True
    num_iters = int(cfg['Dataset']['iteration']['num_iters'])

    # group images which have similar aspect ratio.
    aspect_grouping = cfg['Dataset']['aspect_ratio_factor']

    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle, is_distributed)  # torch.utils.data.sampler.RandomSampler
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        num_workers = nw
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=dataset.collate_fn
        )
        data_loaders.append(data_loader)

    # during training, a single (possibly concatenated) data_loader is returned
    assert len(data_loaders) == 1
    return data_loaders[0]



def building_dataloader_from_cfg(cfg, data_transform):
    nw = cfg['Dataset']['workers'] if cfg['Dataset'].get('workers') else 4
    val_list = cfg['Dataset'].get('val_list')
    val_dataset = make_dataset(
        dataset_list=val_list, transforms=data_transform["val"], is_train=False
    )
    val_data_loader = make_dataloader(cfg, val_dataset, is_train=False, nw=nw)

    if cfg['Dataset'].get('train_s') and cfg['Dataset'].get('train_t'):
        train_s_dataset = make_dataset(
            dataset_list=cfg['Dataset']['train_s'], transforms=data_transform["train"], is_train=True
        )

        train_s_dataloader = make_iter_data_loader(cfg, is_distributed=False, start_iter=0,
                                                  datasets=train_s_dataset, nw=nw)
        train_t_dataset = make_dataset(
            dataset_list=cfg['Dataset']['train_t'], transforms=data_transform["train"], is_train=True
        )

        train_t_dataloader = make_iter_data_loader(cfg, is_distributed=False, start_iter=0,
                                                   datasets=train_t_dataset, nw=nw)

        return train_s_dataloader, train_t_dataloader, val_data_loader
    else:
        if cfg['Dataset'].get('iteration'):
            train_dataset = make_dataset(
                dataset_list=cfg['Dataset']['train_list'], transforms=data_transform["train"], is_train=True
            )

            train_it_loader = make_iter_data_loader(cfg, is_distributed=False, start_iter=0, datasets=train_dataset, nw=nw)

            return train_it_loader, val_data_loader
        else:
            train_dataset = make_dataset(
                dataset_list=cfg['Dataset']['train_list'], transforms=data_transform["train"], is_train=True
            )
            train_data_loader = make_dataloader(cfg, train_dataset, is_train=True, nw=nw)

            return train_data_loader, val_data_loader


def building_backbone(cfg, pretrain=True):
    if cfg['Model']['Backbone'].get('backbone1') == 'fpn':
        backbone = resnet50_fpn_backbone(
            returned_num_layers=4,
            norm_layer=torch.nn.BatchNorm2d,
            frozen_stages=1,
            pretrain=pretrain)
    elif cfg['Model']['Backbone'].get('backbone1') == 'ffm':
        backbone = r50_ffm(
            norm_layer=torch.nn.BatchNorm2d,
            frozen_stages=1,
            adjust_stem=False,
            pretrain=pretrain)
    else:
        backbone = None

    if cfg['Model']['Backbone'].get('backbone2') == 'ffm':
        eg_backbone = r50_ffm(
            norm_layer=torch.nn.BatchNorm2d,
            frozen_stages=1,
            adjust_stem=True,
            pretrain=pretrain)
    else:
        eg_backbone = None

    return backbone, eg_backbone