import torch
import time
import train_utils.distributed_utils as utils
from eva import rpc_evaluation

@torch.no_grad()
def evaluate(model, data_loader, device, output_folder, num=None):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "
    results_dict = {}
    # det_metric = EvalrpcMetric(iou_type="bbox", results_file_name="det_results.json")
    for image, targets, image_ids in metric_logger.log_every(data_loader, 500, header):
        image = list(img.to(device) for img in image)

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()

        outputs = model(image)
        # for io in range(len(outputs)):
        #     for k in list(outputs[io].keys()):
        #         if "masks" in k:
        #             del outputs[io][k]
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        model_time = time.time() - model_time

        # seg_metric.update(targets, outputs)
        metric_logger.update(model_time=model_time)
        # det_metric.update(image_ids, outputs)

        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, outputs)}
        )
    # results_dict = det_metric.synchronize_results()
    image_ids = list(sorted(results_dict.keys()))
    # convert to a list
    predictions = [results_dict[i] for i in image_ids]

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    rpc_info = rpc_evaluation(dataset=data_loader.dataset,
                              predictions=predictions,
                              output_folder=output_folder,
                              iteration=-1,
                              num=num,
                              )

    return rpc_info
