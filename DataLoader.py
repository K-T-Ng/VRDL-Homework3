import os
import sys
import copy
sys.path.append(os.path.join('detectron2-windows'))

import cv2
import torch
import numpy as np
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils

from PrepareDataset import get_nuclei_dicts


def mapper(dataset_dict):
    # avoid in place modification
    dataset_dict = copy.deepcopy(dataset_dict)
    # read image
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    # perform augmentation
    # see detectron2/data/transforms/augmentation_impl.py for more choices
    auginput = T.StandardAugInput(image)
    transforms = auginput.apply_augmentations([
             T.RandomCrop("relative", (0.5, 0.5)),
             T.ResizeShortestEdge(short_edge_length=608,
                                  max_size=800,
                                  sample_style='choice'),
             T.RandomFlip(prob=0.5, horizontal=True, vertical=False)
         ])

    # Obtain the aug image
    image = auginput.image
    image_shape = image.shape[:2]
    dataset_dict["image"] = torch.as_tensor(
        np.ascontiguousarray(image.transpose(2, 0, 1)))

    # update annotations
    annos = [
        utils.transform_instance_annotations(
            annotation, transforms, image_shape)
        for annotation in dataset_dict.pop("annotations")  # dataset_dict["anno
    ]

    instances = utils.annotations_to_instances(
        annos, image_shape, mask_format="bitmask"
    )

    # remove redunant instances
    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict

if __name__ == '__main__':
    ValDicts = get_nuclei_dicts(os.path.join('dataset', 'valid.json'))
