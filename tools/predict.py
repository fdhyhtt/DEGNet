import os
import time
import json
import torch
from train_utils.utils import get_config
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from network_files import FasterRCNN
from train_utils.build import building_backbone
from functions.draw_box_utils import draw_objs


def create_model(num_classes, cfg=None):
    backbone, eg_backbone = building_backbone(cfg)
    model = FasterRCNN(backbone=backbone, eg_backbone=eg_backbone, num_classes=num_classes, cfg=cfg, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main(cfg):
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # create model
    model = create_model(num_classes=201, cfg=cfg)

    # load train weights
    weights_path = "/gemini/ffm_syn_att_da_pds.pth"
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)

    model.load_state_dict(torch.load(weights_path, map_location='cpu')["model"])
    model.to(device)

    # read class_indict
    label_json_path = '../save_files/rpc_indices.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r',) as f:
        class_dict = json.load(f)

    category_index = {str(k): str(v) for k, v in class_dict.items()}

    img_path = "/gemini/data-1/test2019/20180912-14-02-35-410.jpg"
    assert os.path.exists(img_path), "json file {} dose not exist.".format(img_path)
    # load image
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("No targets detected!")

        # font_hanzi = ImageFont.truetype("font/simsun.ttc", 20, encoding="utf-8")

        plot_img = draw_objs(original_img,
                             predict_boxes,
                             predict_classes,
                             predict_scores,
                             category_index=category_index,
                             box_thresh=0.5,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=25,
                             font_hz=None)
        plt.imshow(plot_img)
        plt.show()
        # save
        plot_img.save("/gemini/test_result12.jpg")


if __name__ == "__main__":
    config_path = '../configs/ffm_synins_da_pds.yaml'
    config = get_config(config_path, show=False)
    if not os.path.exists(config['Global']['output_dir']):
        os.makedirs(config['Global']['output_dir'])
    main(config)
