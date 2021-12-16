import os
import sys
sys.path.append(os.path.join('detectron2-windows'))

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.build import build_detection_train_loader
from detectron2 import model_zoo
#from detectron2.engine import DefaultTrainer

from Trainer import Trainer
from DataLoader import mapper
from PrepareDataset import get_nuclei_dicts
from Config import config

if __name__ == "__main__":
    ###############################
    # Register the nuclei dataset #
    ###############################
    TrainJson = os.path.join('dataset', 'train.json')
    ValidJson = os.path.join('dataset', 'valid.json')
    
    TrData_fn = lambda : get_nuclei_dicts(TrainJson)
    TsData_fn = lambda : get_nuclei_dicts(ValidJson)

    DatasetCatalog.register("Nuclei_train", TrData_fn)
    DatasetCatalog.register("Nuclei_valid", TsData_fn)
    
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
