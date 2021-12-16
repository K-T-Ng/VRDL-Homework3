import os
import sys
sys.path.append(os.path.join('detectron2-windows'))

from detectron2.config import get_cfg
from detectron2 import model_zoo


def config():
    yml_path = 'COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml'
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(yml_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yml_path)

    cfg.DATASETS.TRAIN = ("Nuclei_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.BACKBONE.FREEZE_AT = 2
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 28

    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    cfg.SOLVER.MAX_ITER = 100 * 24  # 100 epochs
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 10000
    cfg.SOLVER.WARMUP_ITERS = 3 * 24  # 3 epochs
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (20*24, 50*24, 80*24, 90*24)
    # cfg.SOLVER.REFERENCE_WORLD_SIZE = 1

    # For inference
    # cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 12000
    # cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 2000
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

    cfg.INPUT.MIN_SIZE_TEST = 1000
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.TEST.EVAL_PERIOD = 0
    cfg.TEST.DETECTIONS_PER_IMAGE = 500
    # cfg.TEST.AUG["ENABLED"] = True
    # cfg.TEST.AUG.MIN_SIZES = (1500, 1600, 1700)

    return cfg

if __name__ == '__main__':
    pass
