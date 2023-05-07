import datetime
import math
import sys
import os
from rpctool import evaluate
import torch
import train_utils.distributed_utils as utils
from eva.eval_utils import evaluate


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=500, warmup=False, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch+1}]'
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    lr_scheduler = None
    if epoch == 0 and warmup is True:  # Enable warmup training mode for the first round of training
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)  # mean losses
    for i, [images, targets, _] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Mixed Precision Training Context Manager
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            skip_lr_sched = (scale > scaler.get_scale())
            if not skip_lr_sched:
                if lr_scheduler is not None:
                    lr_scheduler.step()

        else:
            losses.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)

    return mloss, now_lr

def train_by_epoch(args, cfg,  model, optimizer, train_data_loader, val_data_loader, device,
                   scaler=None, print_freq=100, lr_scheduler=None, eva_outdir=None,
                   learning_rate=[], train_loss=[]):
    num = 1
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 10 iterations
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=print_freq, warmup=True,
                                              scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        if (epoch + 1) % 36 == 0:
            # evaluate on the test dataset
            det_info = evaluate(model, val_data_loader, device=device, output_folder=eva_outdir, num=num)
            # write detection into txt
            print(det_info)
            now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            det_results_file = os.path.join(cfg['Global']['output_dir'], f"results_model_{epoch}_{now}.txt")
            with open(det_results_file, "a") as f:
                det_info = str(det_info)
                txt = "epoch:{} {}".format(epoch, ''.join(det_info))
                f.write(txt + "\n")
            num += 1
            # save weights
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            if args.amp:
                save_files["scaler"] = scaler.state_dict()
            torch.save(save_files, os.path.join(cfg['Global']['output_dir'], "model-{}.pth".format(epoch + 1)))
    return train_loss, learning_rate


def do_iter_train(args, cfg, model, optimizer, data_loader, val_data_loader, device,
             checkpoint_period=2500, print_freq=500, scaler=None, lr_scheduler=None,
             eva_outdir=None, learning_rate=[], train_loss=[]):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Training'
    mloss = torch.zeros(1).to(device)  # mean losses
    max_iter = len(data_loader)
    start = args.start_epoch
    cAcc = 0
    for i, [images, targets, _] in enumerate(metric_logger.log_every(data_loader, print_freq, header, start), start):

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Mixed Precision Training Context Manager
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses
        train_loss.append(mloss.item())
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scale = scaler.get_scale()
            
            scaler.update()
            skip_lr_sched = (scale > scaler.get_scale())
            if not skip_lr_sched:
                lr_scheduler.step()
        else:
            losses.backward()
            optimizer.step()
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        learning_rate.append(now_lr)
        metric_logger.update(lr=now_lr)
        
        # val
        iteration = i
        if (iteration+1) == max_iter:
            det_info = evaluate(model, val_data_loader, device=device, output_folder=eva_outdir, num=(iteration+1))
            # write detection into txt
            a_cAcc = det_info['metrics']['cAcc']['averaged']
            now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            det_results_file = os.path.join(cfg['Global']['output_dir'], f"results_model_{iteration+1}_{now}.txt")
            with open(det_results_file, "a") as f:
                det_info = str(det_info)
                txt = "epoch:{} {}".format(iteration+1, ''.join(det_info))
                f.write(txt + "\n")
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'iteration': iteration+1}
            if args.amp:
                save_files["scaler"] = scaler.state_dict()
            torch.save(save_files, os.path.join(cfg['Global']['output_dir'], "model_final.pth"))
        elif (iteration+1) % checkpoint_period == 0 or (iteration+1) in [2]:
            det_info = evaluate(model, val_data_loader, device=device, output_folder=eva_outdir, num=(iteration+1))
            # write detection into txt
            print(det_info)
            a_cAcc = det_info['metrics']['cAcc']['averaged']
            now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            det_results_file = os.path.join(cfg['Global']['output_dir'], f"results_model_{iteration+1}_{now}.txt")
            with open(det_results_file, "a") as f:
                det_info = str(det_info)
                txt = "epoch:{} {}".format(iteration+1, ''.join(det_info))
                f.write(txt + "\n")
            # save weights
            if a_cAcc > cAcc:
                cAcc = a_cAcc
                save_files = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'iteration': iteration+1}
                if args.amp:
                    save_files["scaler"] = scaler.state_dict()
                torch.save(save_files, os.path.join(cfg['Global']['output_dir'], "model-{}.pth".format(iteration+1)))
            model.train()  # restore train state

    return train_loss, learning_rate

def st_train(args, cfg, model, optimizer, data_loader_s, data_loader_t, val_data_loader, device,
             checkpoint_period=2500, scaler=None, print_freq=50, lr_scheduler=None,
             eva_outdir=None, learning_rate=[], train_loss=[]):
    model.train()
    metric_logger = utils.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    mloss = torch.zeros(1).to(device)  # mean losses
    header = 'Training'
    max_iter = len(data_loader_s)
    cAcc = 0
    start = args.start_epoch
    for i, ((images_s, targets_s, _), (images_t, targets_t, _)) in enumerate(metric_logger.log_every_((data_loader_s, data_loader_t), print_freq, header, start), start):
        images = list(image.to(device) for image in images_s+images_t)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets_s+targets_t]

        # Mixed Precision Training Context Manager
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses
        train_loss.append(mloss.item())
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scale = scaler.get_scale()

            scaler.update()
            skip_lr_sched = (scale > scaler.get_scale())
            if not skip_lr_sched:
                lr_scheduler.step()
        else:
            losses.backward()
            optimizer.step()
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        learning_rate.append(now_lr)
        metric_logger.update(lr=now_lr)

        # val
        iteration = i
        if (iteration + 1) == max_iter:
            det_info = evaluate(model, val_data_loader, device=device, output_folder=eva_outdir, num=(iteration + 1))
            # write detection into txt
            a_cAcc = det_info['metrics']['cAcc']['averaged']
            now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            det_results_file = os.path.join(cfg['Global']['output_dir'], f"results_model_{iteration+1}_{now}.txt")
            with open(det_results_file, "a") as f:
                det_info = str(det_info)
                txt = "epoch:{} {}".format(iteration + 1, ''.join(det_info))
                f.write(txt + "\n")
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'iteration': iteration + 1}
            if args.amp:
                save_files["scaler"] = scaler.state_dict()
            torch.save(save_files, os.path.join(cfg['Global']['output_dir'], "model_final.pth"))
        elif ((iteration + 1) % checkpoint_period == 0 and (iteration + 1) in [400000, 540000]):
            det_info = evaluate(model, val_data_loader, device=device, output_folder=eva_outdir, num=(iteration + 1))
            # write detection into txt
            print(det_info)
            a_cAcc = det_info['metrics']['cAcc']['averaged']
            now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            det_results_file = os.path.join(cfg['Global']['output_dir'], f"results_model_{iteration+1}_{now}.txt")
            with open(det_results_file, "a") as f:
                det_info = str(det_info)
                txt = "epoch:{} {}".format(iteration + 1, ''.join(det_info))
                f.write(txt + "\n")
            # save weights
            if a_cAcc > cAcc:
                cAcc = a_cAcc
                save_files = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'iteration': iteration + 1}
                if args.amp:
                    save_files["scaler"] = scaler.state_dict()
                torch.save(save_files, os.path.join(cfg['Global']['output_dir'],"model-{}.pth".format(iteration+1)))
            model.train()  # restore train state
    return train_loss, learning_rate