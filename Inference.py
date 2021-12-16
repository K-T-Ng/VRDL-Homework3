import os
import sys
import json
sys.path.append(os.path.join('detectron2-windows'))

import numpy as np
from pycocotools import mask as COCO_mask
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import detection_utils as utils
from detectron2.data import MetadataCatalog

from Config import config

if __name__ == '__main__':
    # step0: get information from test_img_ids.json
    test_img_ids_path = os.path.join('dataset', 'test_img_ids.json')
    with open(test_img_ids_path, 'r') as f:
        test_img_ids = json.load(f)

    # step1: get the config and set up the predictor
    cfg = config()
    cfg.MODEL.WEIGHTS = os.path.join('weights', 'model_final.pth')
    predictor = DefaultPredictor(cfg)

    # step2: send all images to the predictor and get the predictions
    PredictList = []
    for test_img_dict in test_img_ids:
        image_id = test_img_dict["id"]
        img_path = os.path.join('dataset', 'test', test_img_dict["file_name"])

        img = utils.read_image(img_path, format="BGR")
        out = predictor(img)
        out = out["instances"].to("cpu").get_fields()

        scores = out["scores"].numpy()
        pred_masks = out["pred_masks"].numpy()
        pred_boxes = out["pred_boxes"].tensor.numpy()

        # loop over all instance and pack them into a dictionary
        for score, mask, box in zip(scores, pred_masks, pred_boxes):
            rle = COCO_mask.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode()
            score = float(score)
            box = box.tolist()

            InstanceDict = {
                "image_id": image_id,
                "bbox": box,
                "score": score,
                "category_id": 1,
                "segmentation": rle
            }

            PredictList.append(InstanceDict)

    JSON_OBJ = json.dumps(PredictList, indent=4)
    with open('answer.json', 'w') as f:
        f.write(JSON_OBJ)
