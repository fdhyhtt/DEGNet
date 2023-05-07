import datetime
from train_utils.training_utils import evaluate
import sys
import os
o_path = os.getcwd()
sys.path.append(o_path)
from dataset import transforms
from network_files import FasterRCNN, FastRCNNPredictor
from train_utils.utils import get_config
from train_utils.build import *


def create_model(num_classes, cfg=None):
    backbone, eg_backbone = building_backbone(cfg, pretrain=False)
    model = FasterRCNN(backbone=backbone, eg_backbone=eg_backbone, num_classes=num_classes, cfg=cfg)

    return model


def main(cfg, args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # dataset transform
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    eva_outdir = os.path.join(config['Global']['output_dir'], 'evaout')
    if not os.path.exists(eva_outdir):
        os.makedirs(eva_outdir)

    val_list = cfg['Dataset'].get('val_list')
    val_dataset = make_dataset(
        dataset_list=val_list, transforms=data_transform["val"], is_train=False
    )
    val_data_loader = make_dataloader(cfg, val_dataset, is_train=False, nw=4)

    # create model num_classes equal background + classes
    model = create_model(num_classes=args.num_classes + 1, cfg=cfg)
    model.to(device)

    if not args.cpt:
        print('\nError: No checkpoint_file_path!')
        exit()
    model_files = list(args.cpt)
    if not os.path.exists(eva_outdir):
        os.makedirs(eva_outdir)

    for i, model_file in enumerate(model_files):
        assert os.path.exists(model_file), "not found {} file.".format(model_file)
        model.load_state_dict(torch.load(model_file, map_location='cpu')['model'])
        model.to(device)
        name = list(os.path.split(model_file))[-1]
        name = name.split('.')[0]
        print('Using {} model_file'.format(model_file))

        det_info = evaluate(model, val_data_loader, device=device, output_folder=eva_outdir)
        # write detection into txt
        print(det_info)
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        det_results_file = f"/gemini/output/test_det_results{now}.txt"
        with open(det_results_file, "a") as f:
            det_info = str(det_info)
            txt = "{} {}".format(name, ''.join(det_info))
            f.write(txt + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    # config path
    parser.add_argument('--config', default='../configs/fpn_ins.yaml', help='device')
    # device
    parser.add_argument('--device', default='cuda:0', help='device')
    # classes
    parser.add_argument('--num-classes', default=200, type=int, help='num_classes')
    # checkpoint_file
    parser.add_argument('--cpt', default=None, type=int, help='checkpoint_file_path')
    args = parser.parse_args()
    print(args)
    config = get_config(args.config, show=True)
    if not os.path.exists(config['Global']['output_dir']):
        os.makedirs(config['Global']['output_dir'])
    main(config, args)
