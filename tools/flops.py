import sys
import os
o_path = os.getcwd()
sys.path.append(o_path)
from dataset import transforms
from network_files import FasterRCNN, FastRCNNPredictor
from train_utils.utils import get_config
from train_utils.build import *
from torchvision import transforms as tfs
from ptflops import get_model_complexity_info


def create_model(num_classes, cfg=None):
    backbone, eg_backbone = building_backbone(cfg)
    model = FasterRCNN(backbone=backbone, eg_backbone=eg_backbone, num_classes=num_classes, cfg=cfg)
    return model


def main(cfg, args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # create model num_classes equal background + classes
    model = create_model(num_classes=args.num_classes + 1, cfg=cfg)
    model.to(device)
    flops, params = get_model_complexity_info(model, (3, 800, 800), as_strings=True, print_per_layer_stat=True)
    print('Flops:  ' + flops)
    print('Params: ' + params)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    # config path
    parser.add_argument('--config', default='../configs/ffm_synins_da_pds.yaml', help='device')
    # device
    parser.add_argument('--device', default='cuda:0', help='device')
    # classes
    parser.add_argument('--num-classes', default=200, type=int, help='num_classes')
    # resume path
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # resume from
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # epoch num
    parser.add_argument('--epochs', default=36, type=int, metavar='N',
                        help='number of total epochs to run')
    # mixed precision training
    parser.add_argument("--amp", default=True, help="Use torch.cuda.amp for mixed precision training")
    args = parser.parse_args()
    print(args)
    config = get_config(args.config, show=True)
    if not os.path.exists(config['Global']['output_dir']):
        os.makedirs(config['Global']['output_dir'])
    main(config, args)