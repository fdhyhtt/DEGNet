import sys
import os
o_path = os.getcwd()
sys.path.append(o_path)
from dataset import transforms
from network_files import FasterRCNN, FastRCNNPredictor
from train_utils.utils import get_config
from train_utils.build import *
from train_utils.training_utils import do_iter_train, st_train, train_by_epoch


def create_model(num_classes, cfg=None):
    backbone, eg_backbone = building_backbone(cfg)
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

    # eval result path
    eva_outdir = os.path.join(config['Global']['output_dir'], 'evaout')
    if not os.path.exists(eva_outdir):
        os.makedirs(eva_outdir)

    if cfg['Dataset'].get('train_s') and cfg['Dataset'].get('train_t'):
        train_s_dataloader, train_t_dataloader, val_data_loader = building_dataloader_from_cfg(cfg, data_transform)
    else:
        train_data_loader, val_data_loader = building_dataloader_from_cfg(cfg, data_transform)

    # create model num_classes equal background + classes
    model = create_model(num_classes=args.num_classes + 1, cfg=cfg)
    model.to(device)

    # define optimizer
    optimizer = building_optimizer(cfg, model)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler
    lr_scheduler = make_lr_scheduler(cfg, optimizer)
    # If the weight file from the previous training is specified, continue training with the previous results
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # args.start_epoch = checkpoint['epoch'] + 1
        args.start_epoch = checkpoint['iteration']
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(args.start_epoch))

    val_map = []

    if cfg['Dataset'].get('train_s') and cfg['Dataset'].get('train_t'):
        train_loss, learning_rate = st_train(args, cfg, model, optimizer, train_s_dataloader, train_t_dataloader,
                                             val_data_loader, device, checkpoint_period=10000,  scaler=scaler,
                                             print_freq=100, lr_scheduler=lr_scheduler, eva_outdir=eva_outdir)
    else:
        if not cfg['Dataset'].get('iteration'):
            train_loss, learning_rate = train_by_epoch(args, cfg,  model, optimizer, train_data_loader, val_data_loader,
                                                       device, scaler=scaler, print_freq=100, lr_scheduler=lr_scheduler,
                                                       eva_outdir=eva_outdir)
        else:
            train_loss, learning_rate = do_iter_train(args, cfg, model, optimizer, train_data_loader, val_data_loader,
                                                      device, checkpoint_period=10000, scaler=scaler, print_freq=100,
                                                      lr_scheduler=lr_scheduler, eva_outdir=eva_outdir)

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from functions.plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate, config['Global']['output_dir'])

    # plot mAP curve
    if len(val_map) != 0:
        from functions.plot_curve import plot_map
        plot_map(val_map, config['Global']['output_dir'])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    # config path
    parser.add_argument('--config', default='../configs/ffm_synins.yaml', help='device')
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
