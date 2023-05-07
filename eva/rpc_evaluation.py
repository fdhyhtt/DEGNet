import json
import logging
import os
from datetime import datetime

import boxx
from rpctool import evaluate as eval
from tqdm import tqdm


levels = ['easy', 'medium', 'hard', 'averaged']


def get_cAcc(result, level):
    index = levels.index(level)
    return float(result.loc[index, 'cAcc'].strip('%'))


def check_best_result(output_folder, result, result_str, filename):
    current_cAcc = get_cAcc(result, 'averaged')
    best_path = os.path.join(output_folder, 'best_result.txt')
    if os.path.exists(best_path):
        with open(best_path) as f:
            best_cAcc = float(f.readline().strip())
        if current_cAcc >= best_cAcc:
            best_cAcc = current_cAcc
            with open(best_path, 'w') as f:
                f.write(str(best_cAcc) + '\n' + filename + '\n' + result_str)
    else:
        best_cAcc = current_cAcc
        with open(best_path, 'w') as f:
            f.write(str(current_cAcc) + '\n' + filename + '\n' + result_str)
    return best_cAcc


def rpc_evaluation(dataset, predictions, output_folder, iteration=-1, num=1):
    logger = logging.getLogger("faster_rcnn.val")
    pred_boxlists = []
    annotations = []

    for image_id, prediction in tqdm(enumerate(predictions)):
        img_info = dataset.get_img_info(image_id)

        image_width = img_info["width"]
        image_height = img_info["height"]
        # -----------------------------------------------#
        # -----------------------------------------------#
        # -----------------------------------------------#

        scores = prediction['scores'].numpy().tolist()
        boxs = prediction['boxes'].numpy().tolist()
        labels = prediction['labels'].numpy().tolist()

        for x in range(len(scores)):
            score = scores[x]
            box = boxs[x]
            label = labels[x]
            x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
            pred_boxlists.append({
                "image_id": img_info['id'],
                "category_id": int(label),
                "bbox": [float(k) for k in [x, y, width, height]],
                "score": float(score),
            })

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if len(pred_boxlists) == 0:
        logger.info('Nothing detected.')
        with open(os.path.join(output_folder, 'result_{}.txt'.format(time_stamp)), 'w') as fid:
            fid.write('Nothing detected.')
        return 'Nothing detected.'

    save_path = os.path.join(output_folder, 'bbox_results.json')
    with open(save_path, 'w') as fid:
        json.dump(pred_boxlists, fid)
    res_js = boxx.loadjson(save_path)

    ann_js = boxx.loadjson(dataset.ann_file)
    result = eval(res_js, ann_js, mmap=True)
    logger.info(result)

    result_str = str(result)
    if iteration > 0:
        filename = os.path.join(output_folder, 'result_{:07d}.txt'.format(iteration))
    else:
        if num == None:
            num = 1
        filename = os.path.join(output_folder, f'result-{num}_{time_stamp}.txt')
        num += 1

    with open(filename, 'w') as fid:
        fid.write(result_str)

    best_cAcc = check_best_result(output_folder, result, result_str, filename)
    logger.info('Best cAcc: {}%'.format(best_cAcc))
    metrics = {
        'cAcc': {
            'averaged': get_cAcc(result, 'averaged'),
            'hard': get_cAcc(result, 'hard'),
            'medium': get_cAcc(result, 'medium'),
            'easy': get_cAcc(result, 'easy'),
        }
    }

    eval_result = dict(metrics=metrics)
    return eval_result
