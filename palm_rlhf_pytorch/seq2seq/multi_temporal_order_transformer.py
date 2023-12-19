# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 13:47:01 2023

@author: knechtj
"""

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import pickle
import pandas as pd
# from order_path_dataset import OrderPathDataset
# from order_path_dataset import OrderPathProcessing
from torch.utils.data import DataLoader
import yaml

# NOTE: temporarily setting this globally 

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        # TODO: Eliminate the positional encoding of tgt inputs: 
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)
        
    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# NOTE: We don't want to positionally encode the input sequence, just apply drop-out. 
# TODO: Check that the dim of the output here still makes sense. 
class ApplyDropout(nn.Module):
    def __init__(self,
                 dropout: float):
        self.dropout = nn.Dropout(dropout)
    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding)


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size, padding_idx: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx)
        self.emb_size = emb_size
        self.padding_idx = padding_idx

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PatientEmbedding(nn.Module):
    def __init__(self, input_dim: int, emb_size: int):
        super().__init__()
        self.linear_embedding = nn.Linear(input_dim, emb_size)
        
    def forward(self, x: Tensor):
        # Keep the sqrt() for now. Seems like it applies here too. 
        x = x.to(torch.float32)
        return self.linear_embedding(x)


# token masking
def generate_square_subsequent_mask(device, sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(config, device, src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(device, tgt_seq_len)
    # NOTE: our source mask is FALSE everywhere since the model can attend to the entire set
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)
    src_padding_mask = (src == config["PAD_idx"]).transpose(0, 1)
    tgt_padding_mask = (tgt == config["PAD_idx"]).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# Seq2Seq Network
# This is the standard pytorch transformer model for translation tasks
# Seq2Seq Network
# This is the standard pytorch transformer model for translation tasks
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 d_model: int,
                 seq_length: int, 
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 pat_emb_size: int,
                 pat_input_dim: int, 
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int, 
                 padding_idx: int, 
                 dim_feedforward: int, 
                 dropout: float):
        super(Seq2SeqTransformer, self).__init__()
        self.d_model = d_model
        self.emb_size = emb_size
        self.seq_length = seq_length # NOTE: This is auto-pop. for the standard embeddings given the O-seq-length
        # Need this for the patient X emb. which needs to match the seq-length for repetition 
        self.transformer = Transformer(d_model=d_model,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        # may want a custom embedding class, for now keep this and ensure its learned. 
        self.src_ord_emb = TokenEmbedding(src_vocab_size, 
                                          emb_size, 
                                          padding_idx)

        # fix the embedding for the missing order outcomes to 0 vector (or something)
        self.src_res_emb = TokenEmbedding(src_vocab_size, 
                                          emb_size, 
                                          padding_idx)
        
        # patinput_dim = opd.__getitem__(0)[2][0].long().shape = 841
        self.pat_cov_emb = PatientEmbedding(946, 
                                            pat_emb_size)

        # weighted sum params. are learnable
        self.alpha_o = torch.nn.Parameter(torch.randn(1))
        # self.alpha_o.requires_grad = True
        self.alpha_r = torch.nn.Parameter(torch.randn(1))
        # self.alpha_r.requires_grad = True

        # The target token embeddings stay the same
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, 
                                          emb_size, 
                                          padding_idx)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                orders: Tensor, 
                results: Tensor, 
                pat_cov: Tensor, 
                trg: Tensor
                ):

        print("Order:", orders.shape)
        print("Results:", results.shape)
        # Multimodal embedding: adding each source embedding w. weights from above
        src_emb = torch.add(torch.mul(self.alpha_o, self.src_ord_emb(orders)), 
                            torch.mul(self.alpha_r, self.src_res_emb(results)))  
        print("SRC Emb:", src_emb.shape)
        src_pat_emb = self.pat_cov_emb(pat_cov)
        src_pat_emb = src_pat_emb.unsqueeze(0).repeat(self.seq_length, 1, 1)
        print("PAT Emb:", src_pat_emb.shape)

        src_emb = torch.add(src_emb, src_pat_emb) 
        tgt_emb = self.tgt_tok_emb(trg)
        outs = self.transformer(src_emb, 
                                tgt_emb
                                )
        return self.generator(outs)

    # def encode(self, src_ord: Tensor, src_res: Tensor, src_mask: Tensor):
    #     # Multimodal encoder: 
    #     src_emb = torch.add(torch.mul(self.alpha_o, self.src_ord_emb(src_ord)), 
    #                         torch.mul(self.alpha_r, self.src_res_emb(src_res)))  
    #     return self.transformer.encoder(self.dropout(src_emb), 
    #                                     src_mask)

    # def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
    #     return self.transformer.decoder(self.dropout(self.tgt_tok_emb(tgt)), 
    #                                     memory,
    #                                     tgt_mask)

def greedy_decode(model, src_ord, src_res, pat_cov, max_len, BOS_idx, EOS_idx):
    model.eval()

    src_pat_emb = model.pat_cov_emb(pat_cov)
    src_pat_emb.unsqueeze(0).repeat(max_len, 1, 1)
    src_emb = torch.add(torch.mul(model.alpha_o, model.src_ord_emb(src_ord)), 
                        torch.mul(model.alpha_r, model.src_res_emb(src_res)))  
    src_emb = torch.add(src_emb, src_pat_emb) # (80, 1, 256)
    print(f"src_emb: {src_emb.shape}") 
    memory = model.transformer.encoder(src_emb) # (80, 1, 256)
    print(f"memory: {memory.shape}") 

    # Start the output tensor with the start symbol
    ys = torch.ones(1, 1).fill_(BOS_idx).type_as(src_ord.data)
    print(f"ys: {ys.shape}")

    for i in range(max_len-1):
        #Decode one step at a time 
        tgt_emb = model.tgt_tok_emb(ys)
        print(f"tgt_emb: {tgt_emb.shape}")
        out = model.transformer.decoder(tgt_emb, memory)
        out = out.transpose(0,1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        #Append the predicted word to the output sequence
        ys = torch.cat([ys, torch.ones(1, 1).fill_(next_word).type_as(src_ord.data)], dim=0)
        print(f"iteration{i}: {ys.shape}")

        #Break if EOS token is predicted
        if next_word == EOS_idx: 
            break
    return ys 

if __name__ == "__main__":
    if torch.cuda.is_available():
        model_device = torch.device('cuda')
    else:
        model_device = torch.device('cpu')

    with open('config_008.yaml') as file: 
        config = yaml.safe_load(file)
    
    model_params = config['model_params']
    model = Seq2SeqTransformer(d_model=model_params['MODEL_DIM'], 
                               seq_length=config['max_seq_length'],
                               num_encoder_layers=model_params['NUM_ENCODER_LAYERS'],
                               num_decoder_layers=model_params['NUM_DECODER_LAYERS'],
                               emb_size=model_params['EMB_SIZE'],
                               pat_emb_size=model_params['EMB_SIZE'], #pat_emb_size is same as emb_size since we need the tensors to be the same size to get an element-wise sum  
                               pat_input_dim=model_params['PAT_INPUT_DIM'],
                               nhead=model_params['NHEAD'],
                               src_vocab_size=model_params['SRC_VOCAB_SIZE'],
                               tgt_vocab_size=model_params['TGT_VOCAB_SIZE'], 
                               padding_idx=config['PAD_idx'],
                               dim_feedforward=model_params['FFN_HID_DIM'],
                               dropout=model_params['DROPOUT']
                                
    )

    # Load the saved model weights
    checkpoint = torch.load('transformer_0.0.8.pth', map_location=model_device)
    model.load_state_dict(checkpoint['model_state_dict'])

    #set model to eval mode
    model.eval()

    # Load batch tensor
    batch_data = torch.load('batch_tensor.pt')

    orders = batch_data[0][0].unsqueeze(1)  # First tensor, add batch dim #(80,1)
    results = batch_data[1][0].unsqueeze(1)  # Second tensor, add batch dim  #(80,1)
    pat_cov = batch_data[2][0] # Third tensor, add batch dim #(946, 1)
    trg = batch_data[3][0].unsqueeze(1)     # Fourth tensor, add batch dim  #(80,1)


    # straight up forward pass TODO: delete this later 
    # output = model(
    #     orders, results, pat_cov, trg
    # )
    # print(output) 
    # print(f"output shape: {output.shape}") #(1, 80, 7783)

    # Greedy decoding 
    decoded = greedy_decode(model, 
                            orders, 
                            results, 
                            pat_cov, 
                            max_len=orders.shape[0], 
                            BOS_idx=config['BOS_idx'],
                            EOS_idx=config['EOS_idx'])

    print(decoded)

