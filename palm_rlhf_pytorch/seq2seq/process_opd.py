from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import pickle
import yaml
import pandas as pd
from order_path_dataset import OrderPathDataset
from order_path_dataset import OrderPathProcessing
from torch.utils.data import DataLoader


def main():
    # Loading config: 
    print("...Init...")
    with open("/wynton/protected/home/rizk-jackson/jknecht/order_path_prediction/experiments/scripts/config_006.yaml", "r") as f:
        config = yaml.safe_load(f)
    path = config["path"] 
    torch.manual_seed(config["seed"])
    
    # Process order-path-data, this takes a minute. 
    print("...Processing Data...")
    opd_processed = OrderPathProcessing(config, path)
    torch.save(opd_processed, (config["data_loader_path"] + "opd_processed.pt"))
    train_val_test = opd_processed._split_encounters(seed = config["seed"])

    print("...Init OPD...")
    # Init. datasets for train/val:
    opd_train = OrderPathDataset(config = config, 
                                 path = path, 
                                 order_df = opd_processed._order_data, 
                                 results_df = opd_processed._outcomes_data, 
                                 patient_df = opd_processed._patient_data,
                                 encounter_ids = train_val_test["train_encounter_ids"])
    # Collating encounters and returning training df: 
    opd_train._collate_encounter_orders()
    opd_train._collate_encounter_results()
    opd_train._collate_encounter_patient_data()
    torch.save(opd_train, (config["data_loader_path"] + "opd_train.pt"))

    opd_val = OrderPathDataset(config = config, 
                               path = path, 
                               order_df = opd_processed._order_data, 
                               results_df = opd_processed._outcomes_data, 
                               patient_df = opd_processed._patient_data,
                               encounter_ids = train_val_test["val_encounter_ids"])
    # Collating encounters and returning training df: 
    opd_val._collate_encounter_orders()
    opd_val._collate_encounter_results()
    opd_val._collate_encounter_patient_data()
    torch.save(opd_val, (config["data_loader_path"] + "opd_val.pt"))
    
    print("...End...")
    return 

    
if __name__ == "__main__":
           main()





