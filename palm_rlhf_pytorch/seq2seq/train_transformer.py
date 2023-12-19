from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import pickle
import yaml
import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from order_path_dataset import OrderPathDataset
from order_path_dataset import OrderPathProcessing
from multi_temporal_order_transformer import Seq2SeqTransformer
from multi_temporal_order_transformer import create_mask
from multi_temporal_order_transformer import generate_square_subsequent_mask


# Training: 
# function to collate data samples into batch tensors
def collate_batch_fn(batch):
    src_order_batch, tgt_order_batch, src_result_batch, src_pat_batch = [], [], [], []
    
    for pair in batch: 
        src_order_batch.append(pair[0][0])
        tgt_order_batch.append(pair[0][1])
        src_result_batch.append(pair[1][0])
        src_pat_batch.append(pair[2][0])
        
    return (src_order_batch, src_result_batch, src_pat_batch, tgt_order_batch)

def _get_path_tensors(batch):
    # pairing is handled by the data-loader
    # we abstract away from the encounter level so now its a single call 
    src_order_batch = batch[0] 
    src_result_batch = batch[1]
    src_pat_batch = batch[2]
    tgt_order_batch = batch[3]
    return src_order_batch, src_result_batch, src_pat_batch, tgt_order_batch

def train_mm_epoch(config, device, model, loss_fn, optimizer, train_dataloader):
    model.train()
    losses = 0

    # Returns single batch at a time: 
    for batch in tqdm.tqdm(train_dataloader, desc="...Train batches: ", ascii="░▒█"):
        # NOTE: need to collect src and tgt order sets for the batch here: 
        src_ord, src_res, src_pat, tgt_ord = _get_path_tensors(batch)
        # need to return 'results' from collate_fn() in the batch 
        # collect src and tgt again since we have two layers (i.e. multiple sequences per batch) 
        # need to collect src and tgt into simple lists
        src_ord = torch.transpose(torch.stack(src_ord), 0,1)
        src_res = torch.transpose(torch.stack(src_res), 0,1)
        src_pat = torch.stack(src_pat)
        tgt_ord = torch.transpose(torch.stack(tgt_ord), 0,1)
        # cast to GPU
        src_ord = src_ord.to(device)
        src_res = src_res.to(device)
        src_pat = src_pat.to(device)
        tgt_ord = tgt_ord.to(device)
        # need to shift the target by one. This is more complicated for me. 
        tgt_input = tgt_ord[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(config, device, src_ord, tgt_input)
        logits = model(src_ord,
                       src_res, 
                       src_pat, 
                       tgt_input, 
                       src_mask, 
                       tgt_mask,
                       src_padding_mask, 
                       tgt_padding_mask, 
                       src_padding_mask)
        optimizer.zero_grad()
        tgt_out = tgt_ord[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()
    return losses / len(list(train_dataloader))


def evaluate_mm_transformer(config, device, model, loss_fn, val_dataloader):
    model.eval()
    losses = 0

    for batch in tqdm.tqdm(val_dataloader, desc="...Val Batches: ", ascii="░▒█"):
        src_ord, src_res, src_pat, tgt_ord = _get_path_tensors(batch)
        src_ord = torch.transpose(torch.stack(src_ord), 0,1)
        src_res = torch.transpose(torch.stack(src_res), 0,1)
        src_pat = torch.stack(src_pat)
        tgt_ord = torch.transpose(torch.stack(tgt_ord), 0,1)
        src_ord = src_ord.to(device)
        src_res = src_res.to(device)
        src_pat = src_pat.to(device)
        tgt_ord = tgt_ord.to(device)
        tgt_input = tgt_ord[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(config, device, src_ord, tgt_input)
        logits = model(src_ord,
                       src_res, 
                       src_pat, 
                       tgt_input, 
                       src_mask, 
                       tgt_mask,
                       src_padding_mask, 
                       tgt_padding_mask, 
                       src_padding_mask)
        tgt_out = tgt_ord[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))


def get_unique_orders(order_df):
    # Returns unique order on id
    return order_df.dropna(subset=['encounter_id', 'order_types_id']).order_types_id.unique()


def process_loss(train_loss, val_loss):
    loss_df = pd.DataFrame({"epochs": list(range(1,len(train_loss)+1)),
                            "train_loss": train_loss, 
                            "val_loss": val_loss})
    return loss_df


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least, then save the
    model state.
    """
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        
    def __call__(self, config, current_valid_loss, epoch, model, optimizer, criterion):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': criterion,
                       },config["best_model_path"])


def main():
    device = torch.device('cuda') 
    # Loading config: 
    print("...Init...")
    with open("/wynton/protected/home/rizk-jackson/jknecht/order_path_prediction/experiments/scripts/config_008.yaml", "r") as f:
        config = yaml.safe_load(f)
    path = config["path"] 
    torch.manual_seed(config["seed"])
    print("...Device is set to: ", device)
    
    # Process order-path-data, this takes a minute. 
    print("...Processing Data...")
    opd_processed = torch.load((config["data_loader_path"] + "opd_processed.pt"))
    opd_train = torch.load((config["data_loader_path"] + "opd_train.pt"))
    opd_val = torch.load((config["data_loader_path"] + "opd_val.pt"))

    # print("...TEST: Init OPD...")
    # NOTE: this is to truncate the train-val IDs. Otherwise take the full opd's from above. 
    # Init. datasets for train/val:
    # train_val_test = opd_processed._split_encounters(seed = config["seed"])
    # opd_train = OrderPathDataset(config = config, 
    #                              path = path, 
    #                              order_df = opd_processed._order_data, 
    #                              results_df = opd_processed._outcomes_data, 
    #                              patient_df = opd_processed._patient_data,
    #                              encounter_ids = train_val_test["train_encounter_ids"][1:10])
    # opd_train._collate_encounter_orders()
    # opd_train._collate_encounter_results() 
    # opd_val = OrderPathDataset(config = config, 
    #                            path = path, 
    #                            order_df = opd_processed._order_data, 
    #                            results_df = opd_processed._outcomes_data, 
    #                            patient_df = opd_processed._patient_data,
    #                            encounter_ids = train_val_test["val_encounter_ids"][1:10])
    # opd_val._collate_encounter_orders()
    # opd_val._collate_encounter_results()
    
    # Init model:     
    _order2idx = opd_train._order2idx
    _idx2order = opd_train._idx2order
    unique_orders = opd_processed.unique_orders # From the processed data
    SRC_VOCAB_SIZE = len(_order2idx)
    TGT_VOCAB_SIZE = len(_order2idx)
    
    # Model params from config: 
    SEQ_LENGTH = config["max_seq_length"]
    MODEL_DIM = config["model_params"]["MODEL_DIM"]
    EMB_SIZE = config["model_params"]["EMB_SIZE"]
    NHEAD = config["model_params"]["NHEAD"]
    FFN_HID_DIM = config["model_params"]["FFN_HID_DIM"]
    BATCH_SIZE = config["model_params"]["BATCH_SIZE"]
    NUM_ENCODER_LAYERS = config["model_params"]["NUM_ENCODER_LAYERS"]
    NUM_DECODER_LAYERS = config["model_params"]["NUM_DECODER_LAYERS"]
    DROPOUT = config["model_params"]["DROPOUT"]
    
    # Define special symbols and indices
    PAD_IDX = config["PAD_idx"]
    BOS_IDX = config["BOS_idx"]
    EOS_IDX = config["EOS_idx"]
    SEP_IDX = config["SEP_idx"] # Otherwise this is 4 
    UNK_IDX = config["UNK_idx"]
    
    transformer = Seq2SeqTransformer(d_model = MODEL_DIM, 
                                     seq_length = SEQ_LENGTH,
                                     num_encoder_layers = NUM_ENCODER_LAYERS, 
                                     num_decoder_layers = NUM_DECODER_LAYERS, 
                                     emb_size = EMB_SIZE,
                                     pat_emb_size = EMB_SIZE, 
                                     pat_input_dim = 946, 
                                     nhead = NHEAD, 
                                     src_vocab_size = SRC_VOCAB_SIZE,
                                     tgt_vocab_size = TGT_VOCAB_SIZE, 
                                     padding_idx = PAD_IDX, 
                                     dim_feedforward = FFN_HID_DIM,
                                     dropout = DROPOUT)
    transformer = transformer.to(device)
    
    # Applyung U-transform to high-dim parameters: 
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    # Loss and g-optimizer: 
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(transformer.parameters(), 
                                 lr=config["model_params"]["LR"], # increase this a bit
                                 betas=(0.9, 0.98), 
                                 eps=1e-9)

    print("...Init train/val dataloader...")
    train_dataloader = DataLoader(opd_train, 
                                  batch_size=BATCH_SIZE, 
                                  collate_fn=collate_batch_fn)
    
    val_dataloader = DataLoader(opd_val, 
                                batch_size=BATCH_SIZE, 
                                collate_fn=collate_batch_fn)

    
    # Training loop: 
    print("...Starting training loop...")
    save_best_model = SaveBestModel()
    train_loss = []
    val_loss = []
    for epoch in range(1, config["num_epochs"]+1):
        
        train_epoch_loss = train_mm_epoch(config = config,
                                          device = device, 
                                          model = transformer, 
                                          loss_fn = loss_fn, 
                                          optimizer = optimizer, 
                                          train_dataloader = train_dataloader)
        train_loss.append(train_epoch_loss)
        
        val_epoch_loss = evaluate_mm_transformer(config = config, 
                                                 device = device,
                                                 model = transformer, 
                                                 loss_fn = loss_fn,
                                                 val_dataloader = val_dataloader)
        val_loss.append(val_epoch_loss)
        
        print(f"Epoch: {epoch}, Train loss: {train_epoch_loss:.3f}, Val loss: {val_epoch_loss:.3f}")
    
        save_best_model(config = config, 
                        current_valid_loss = val_epoch_loss, 
                        epoch = epoch, 
                        model = transformer,
                        optimizer = optimizer, 
                        criterion = loss_fn)

        print("...Updating loss df...")
        loss_df = process_loss(train_loss, val_loss)
        loss_df.to_csv(config["loss_df_path"], index=False,  mode='w')
        
    print("...End...")
    return 

    
if __name__ == "__main__":
           main()










