# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""
import errno
import pkgutil
import shutil
import sys
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen, Request
import torch
import os
from importlib import import_module

import torchvision
from tqdm import tqdm


ENV_TORCH_HOME = 'TORCH_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'


class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        # -------------------------------------------------
        # -----------------RPC eval dataset----------------
        # -------------------------------------------------
        "rpc_2019_test": {
            "image_dir": r"/gemini/data-1/test2019",
            "ann_file": r'/gemini/data-1/instances_test2019.json',
        },
        # -------------------------------------------------
        # -----------------RPC configs----------------
        # -------------------------------------------------

        "rpc_train_syn": {
            "image_dir": "/gemini/data-3/syn_images",
            "ann_file": '/gemini/instances_syn.json',
            'rendered': False,
        },
        "rpc_instance": {
            "image_dir": r"/gemini/data-1/val2019",
            "ann_file": r'/gemini/data-1/instances_val2019.json',
            "is_train": True,
        },
        "rpc_train_render": {
            "image_dir": "/gemini/data-2/render",
            "ann_file": '/gemini/instances_syn.json',
            'rendered': True,
        },

        "rpc_source": {
            "image_dir": r"/gemini/data-1/train2019",
            "ann_file": r'/gemini/data-1/instances_train2019.json',
            "is_train": True,
        },
    }

    @staticmethod
    def get(name):
        if "rpc_train" in name:
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(attrs)
            return dict(
                factory="RPCDatasetTrain",
                args=args,
            )
        elif name in ('rpc_instance',):
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(attrs)
            return dict(
                factory="RPCTestDataset",
                args=args,
            )
        elif name in ('rpc_2019_test',):
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(attrs)
            return dict(
                factory="RPCTestDataset",
                args=args,
            )
        elif name in ('rpc_source', ):
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(attrs)
            return dict(
                factory="RPCTestDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


def download_url_to_file(url, dst, progress=True):
    r"""Download object at the given URL to a local path.

    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True

    Example:
        >>> torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')

    """
    file_size = None
    # We use a different API for python2 since urllib(2) doesn't recognize the CA
    # certificates in older Python
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    print(dst_dir)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                pbar.update(len(buffer))

        f.close()
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


def _get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(ENV_TORCH_HOME,
                  os.path.join(os.getenv(ENV_XDG_CACHE_HOME,
                                         DEFAULT_CACHE_DIR), 'torch')))
    return torch_home


def check_pretrain_file():
    # catalog lookup
    model_url = get_torchvision_models()
    url = model_url['resnet50']
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    model_dir = os.path.join(_get_torch_home(), 'hub')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        download_url_to_file(url, cached_file, progress=True)
    return cached_file


def get_torchvision_models():
    model_urls = dict()
    for _, name, ispkg in pkgutil.walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = import_module(f'torchvision.models.{name}')
        if hasattr(_zoo, 'model_urls'):
            _urls = getattr(_zoo, 'model_urls')
            model_urls.update(_urls)
    return model_urls

