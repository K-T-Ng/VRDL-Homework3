import os
import sys

from Trainer import Trainer
from DataLoader import mapper
from PrepareDataset import get_nuclei_dicts
from Config import config

sys.path.append(os.path.join('detectron2-windows'))
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.build import build_detection_train_loader
from detectron2 import model_zoo

if __name__ == "__main__":
    ###############################
    # Register the nuclei dataset #
    ###############################
    TrainJson = os.path.join('dataset', 'train.json')
    ValidJson = os.path.join('dataset', 'valid.json')

    def TrData_fn():
        return get_nuclei_dicts(TrainJson)

    DatasetCatalog.register("Nuclei_train", TrData_fn)

    ##############
    # Get config #
    ##############
    cfg = config()

    #################
    # Setup trainer #
    #################
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
