import os
import json
import random
from collections import defaultdict
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from scipy import ndimage


class RPCDatasetTrain(Dataset):
    def __init__(self, image_dir, ann_file, transforms=None, rendered=False):
        super().__init__()
        # image path
        assert os.path.exists(image_dir), "path '{}' does not exist.".format(image_dir)
        self.image_dir = image_dir
        # ann_file path
        with open(ann_file, 'r') as f:
            data = json.load(f)
        self.data = data
        self.rendered = rendered
        file_name = []
        for image in data:
            file_name.append(image['image_id'])
        images_path = [os.path.join(image_dir, x) for x in file_name]
        self.file_name = file_name
        self.images_path = images_path

        self.scale = 1.0
        self.ext = '.jpg'
        self.image_size = 800 if self.rendered else 1824
        if self.rendered:  # Rendered image is 800*800 and format is png
            self.scale = 800.0 / 1824.0
            self.ext = '.png'

        self.transforms = transforms

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        ann = self.data[idx]
        image_id = ann['image_id']
        image_name = os.path.splitext(image_id)[0]
        img_path = os.path.join(self.image_dir, image_name + self.ext)

        assert os.path.exists(img_path), f"not find {img_path}"

        img = Image.open(img_path).convert('RGB')
        image_size = self.image_size
        assert img.width == image_size
        assert img.height == image_size
        target = self.parse_objects(idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def __len__(self):
        return len(self.images_path)

    def get_height_and_width(self):
        if self.rendered:
            return 800, 800
        return 1824, 1824

    def get_img_info(self):
        if self.rendered:
            return {"height": 800, "width": 800}
        return {"height": 1824, "width": 1824}

    def get_annotation(self, image_id):
        ann = self.data[image_id]
        return ann

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    def parse_objects(self, idx: int):
        """
        return results
        """
        boxes = []
        labels = []
        iscrowd = []
        data = self.data[idx]
        for obj in data["objects"]:
            xmin = float(obj["bbox"][0])
            ymin = float(obj["bbox"][1])
            w = float(obj["bbox"][2])
            h = float(obj["bbox"][3])
            xmax = xmin + w
            ymax = ymin + h
            # check data
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(data['image_id']))
                continue
            boxes.append([xmin * self.scale, ymin * self.scale, xmax * self.scale, ymax * self.scale])
            labels.append(obj["category_id"])
            iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        return {"boxes": boxes,
                "labels": labels,
                "iscrowd": iscrowd,
                "image_id": image_id,
                "area": area}


class RPCTestDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, ann_file, is_train=False, transforms=None):
        self.transforms = transforms
        # image path
        assert os.path.exists(image_dir), "path '{}' does not exist.".format(image_dir)
        self.images_dir = image_dir
        assert os.path.exists(ann_file), "file '{}' does not exist.".format(ann_file)
        self.ann_file = ann_file
        with open(self.ann_file) as fid:
            data = json.load(fid)

        annotations = defaultdict(list)
        images = []
        for image in data['images']:
            images.append(image)
        for ann in data['annotations']:
            bbox = ann['bbox']
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            annotations[ann['image_id']].append((ann['category_id'], x, y, w, h))
        images_name = []
        for x in images:
            name = x['file_name']
            images_name.append(name)
        self.images_name = images_name
        self.images = images
        self.annotations = dict(annotations)
        self.is_train = is_train
        self.coco = COCO(self.ann_file)

        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # image_id = self.images[index]['id']
        # img_path = os.path.join(self.images_dir, self.images[index]['file_name'])
        # img = Image.open(img_path).convert("RGB")
        # width, height = img.size[0], img.size[1]
        # boxes = []
        # labels = []
        # ann = self.annotations[image_id]
        # for category, x, y, w, h in ann:
        #     boxes.append([x, y, x + w, y + h])
        #     labels.append(category)
        if self.is_train:
            coco = self.coco
            img_id = self.ids[index]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            coco_target = coco.loadAnns(ann_ids)

            img_path = os.path.join(self.images_dir, self.images[index]['file_name'])
            img = Image.open(img_path).convert("RGB")

            w, h = img.size
            target = self.parse_targets(img_id, coco_target, w, h)
        else:
            img_path = os.path.join(self.images_dir, self.images[index]['file_name'])
            img = Image.open(img_path).convert("RGB")
            target = None
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def get_annotation(self, image_id):
        ann = self.annotations[image_id]
        return ann

    def parse_targets(self,
                      img_id: int,
                      coco_targets: list,
                      w: int = None,
                      h: int = None):
        assert w > 0
        assert h > 0

        anno = [obj for obj in coco_targets if obj['iscrowd'] == 0]
        boxes = [obj["bbox"] for obj in anno]

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # [xmin, ymin, w, h] -> [xmin, ymin, xmax, ymax]
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])

        segmentations = [obj["segmentation"] for obj in anno]
        # masks = convert_coco_poly_mask(segmentations, h, w)

        # x_max>x_min and y_max>y_min
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        # masks = masks[keep]
        area = area[keep]
        iscrowd = iscrowd[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        # target["masks"] = masks
        target["image_id"] = torch.tensor([img_id])

        # for conversion to coco api
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target

    def __len__(self):
        return len(self.images)

    def get_img_info(self, index):
        image = self.images[index]
        return {"height": image['height'], "width": image['width'], "id": image['id'], 'file_name': image['file_name']}

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
