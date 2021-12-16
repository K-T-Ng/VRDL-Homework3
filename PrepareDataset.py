import os
import sys
import json
sys.path.append(os.path.join('detectron2-windows'))

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

def generate_train_json(dataset_loc=os.path.join('dataset', 'train'),
                        save_loc=os.path.join('dataset', 'train.json')):
    NameList = os.listdir(dataset_loc)
    dataset_dicts = []

    for img_id, img_name in enumerate(NameList):
        record = {}

        # file_name: the full path to the image file
        record["file_name"] = os.path.join(dataset_loc, img_name,
                                           'images', img_name+'.png')

        # height, width: integer. The shape of the image
        img = Image.open(record["file_name"])
        record["height"] = img.size[0]
        record["width"] = img.size[1]

        # image_id: a unique id that identifies this image
        record["image_id"] = img_id

        # annotations(list[dict])
        MaskList = os.listdir(os.path.join(dataset_loc, img_name, 'masks'))
        annotations = []
        for msk_name in MaskList:
            # a mask's suffix should be '.png'
            if not msk_name.endswith('.png'):
                continue
            
            obj = {}
            # read the mask and convert to binary ndarray
            msk = cv2.imread(os.path.join(dataset_loc, img_name, 'masks',
                                          msk_name))
            msk = np.sum(msk, axis=2) > 0
            msk = np.asfortranarray(msk)

            # category_id: an integer in range [0, num_cat-1]
            # num_cat is reserved to represent background
            obj["category_id"] = 0

            # segmentation: rle object from pycocotools.mask.encode()
            obj["segmentation"] = mask.encode(msk)
            obj["segmentation"]["counts"] = obj["segmentation"]["counts"].decode()

            # bbox_mode: the format of bbox (xywh here)
            obj["bbox_mode"] = BoxMode.XYWH_ABS

            # bbox(list[float])
            obj["bbox"] = mask.toBbox(obj["segmentation"]).tolist()

            annotations.append(obj)
        record["annotations"] = annotations
        dataset_dicts.append(record)
        print(img_id, img_name)

    JSON_OBJ = json.dumps(dataset_dicts)
    with open(save_loc, 'w') as f:
        f.write(JSON_OBJ)
    return

def get_nuclei_dicts(json_loc=os.path.join('data', 'train.json')):
    with open(json_loc, 'r') as f:
        dataset_dicts = json.load(f)

    # set each annotation["bbox_mode"] to BoxMode.XYWH_ABS
    for data in dataset_dicts:
        for anno in data["annotations"]:
            anno["bbox_mode"] = BoxMode.XYWH_ABS

    return dataset_dicts

if __name__ == '__main__':
    GENERATE_JSON = False
    VISUALIZE = True

    ####################################
    # Generate train & valid json here #
    ####################################
    if GENERATE_JSON:
        generate_train_json(dataset_loc=os.path.join('dataset', 'train'),
                            save_loc=os.path.join('dataset', 'train.json'))
        
        generate_train_json(dataset_loc=os.path.join('dataset', 'valid'),
                            save_loc=os.path.join('dataset', 'valid.json'))
    

    #############################
    # Test for register dataset #
    #############################
    TsData_fn = lambda : get_nuclei_dicts(json_loc=os.path.join('dataset',
                                                                'valid.json'))

    DatasetCatalog.register("Nuclei_valid", TsData_fn)
    dataset_dicts = DatasetCatalog.get("Nuclei_valid")
    MetadataCatalog.get("Nuclei_valid").set(thing_classes=["Nuclei"])
    Nuclei_metadata = MetadataCatalog.get("Nuclei_valid")
    
    ##################
    # Visualize data #
    ##################
    if VISUALIZE:
        for data in dataset_dicts:
            for data_key, data_val in data.items():
                if data_key != "annotations":
                    print(data_key, data_val)
                else:
                    anno = data_val[0]
                    for anno_key, anno_val in anno.items():
                        print(anno_key, anno_val)
        
        img = cv2.imread(data["file_name"])
        visualizer = Visualizer(img[:, :, ::-1])
        out = visualizer.draw_dataset_dict(data)
        cv2.imshow('', out.get_image()[:, :, ::-1])
        cv2.waitKey(-1)
