import os
import sys
sys.path.append(os.path.join('dataset', 'detectron2-windows'))

from detectron2.data.build import build_detection_train_loader
from detectron2.engine.defaults import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from DataLoader import mapper

class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper)
