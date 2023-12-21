'''
https://pytorch.org/tutorials/beginner/translation_transformer.html
'''

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import pickle
import pandas as pd
from torch.utils.data import DataLoader
import yaml


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
                 dropout: float, 
                 BOS_idx: int, 
                 EOS_idx: int):
        super(Seq2SeqTransformer, self).__init__()
        self.d_model = d_model
        self.emb_size = emb_size
        self.seq_length = seq_length 
        self.transformer = Transformer(d_model=d_model,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout
                                       )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

        #Embeddings: 
        self.src_ord_emb = TokenEmbedding(src_vocab_size, 
                                          emb_size, 
                                          padding_idx)

        self.src_res_emb = TokenEmbedding(src_vocab_size, 
                                          emb_size, 
                                          padding_idx)
        
        self.pat_cov_emb = PatientEmbedding(946, 
                                            pat_emb_size)

        self.alpha_o = torch.nn.Parameter(torch.randn(1))
        self.alpha_r = torch.nn.Parameter(torch.randn(1))

        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, 
                                          emb_size, 
                                          padding_idx)

        self.dropout = nn.Dropout(dropout)
        self.padding_idx = padding_idx
        self.BOS_idx = BOS_idx
        self.EOS_idx = EOS_idx

    # def preprocess(self, orders, results, pat_cov, trg):
    #     src_emb = torch.add(torch.mul(self.alpha_o, self.src_ord_emb(orders)), 
    #                         torch.mul(self.alpha_r, self.src_res_emb(results)))  
    #     src_pat_emb = self.pat_cov_emb(pat_cov) #shape: (1, 256)
    #     src_pat_emb = src_pat_emb.unsqueeze(1).repeat(1, self.seq_length, 1)
    #     src_emb = torch.add(src_emb, src_pat_emb)

    #     trg_emb = self.tgt_tok_emb(trg)
    #     return src_emb

    def forward(self,
                orders: Tensor, 
                results: Tensor, 
                pat_cov: Tensor, 
                trg: Tensor
                ):
        # self.preprocess(orders, results, pat_cov, trg)
        src_emb = torch.add(torch.mul(self.alpha_o, self.src_ord_emb(orders)), 
                            torch.mul(self.alpha_r, self.src_res_emb(results)))  
        src_pat_emb = self.pat_cov_emb(pat_cov) #shape: (1, 256)
        src_pat_emb = src_pat_emb.unsqueeze(1).repeat(1, self.seq_length, 1)
        src_emb = torch.add(src_emb, src_pat_emb)

        trg_emb = self.tgt_tok_emb(trg)
        
        #I'm not sure if this trg_emb padding is needed or not.. (#TODO: needs modification here)
        # if self.src_emb.size(1) != self.trg_emb.size(1):
        #     trg_emb= torch.nn.functional.pad(trg_emb, (0, 0, 0, 79), value=self.padding_idx)
        print(f"trg_emb shape: {trg_emb.shape}")
        outs = self.transformer(src=src_emb, 
                                tgt=trg_emb
                                )
                                
        return self.generator(outs) # output size = torch.Size([1, 80, 7783])

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def create_padding_mask(self, seq):
        return (seq == self.padding_idx) #hard coding PAD-IDX token for now
    
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
                               dropout=model_params['DROPOUT'], 
                               BOS_idx=config['BOS_idx'], 
                               EOS_idx=config['EOS_idx']
                                
    )

    # print(f"model.dim: {model.d_model}")
    print(f"model.parameters: {model.parameters()}")
    # Load the saved model weights
    checkpoint = torch.load('transformer_0.0.8.pth', map_location=model_device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load batch tensor
    batch_data = torch.load('batch_tensor.pt')

    # orders = batch_data[0][0].unsqueeze(0)  # First tensor, add batch dim #(1,80)
    # results = batch_data[1][0].unsqueeze(0)  # Second tensor, add batch dim  #(1,80)
    # pat_cov = batch_data[2][0].unsqueeze(0) # Third tensor, add batch dim #(1,946)
    # trg = batch_data[3][0].unsqueeze(0)     # Fourth tensor, add batch dim  #(1,80)

    print(f"adding second dimension at the back: {batch_data[0][0].unsqueeze(1)}")
    # If we try to do autoregressive generation: - #TODO: comment this out when finished testing
    # trg = torch.tensor([[config['BOS_idx']]], device=model_device)
    # print(f"printing start_trg: {trg}, trg_shape: {trg.shape}")

    output = model(
        orders, results, pat_cov, trg
    )
    print(output) 
    print(f"output shape: {output.shape}") #(1, 80, 7783)

    # Greedy decoding 
