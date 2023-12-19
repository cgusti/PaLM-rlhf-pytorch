'''
Train an LLM using reward signals from a reward model via PPO 
Author: Claudia Gusti
'''

import torch 
from seq2seq.multi_temporal_order_transformer import Seq2SeqTransformer
from seq2seq.reward_model import LinearRewardModel
from palm_rlhf_pytorch.ppo import RLHFTrainer


if torch.cuda.is_available():
    model_device = torch.device('cuda')
else:
    model_device = torch.device('cpu')

#Load pretrained seq2seq transformer 
with open('seq2seq/config_008.yaml') as file: 
    config = yaml.safe_load(file)

model_params = config['model_params']
checkpoint = torch.load('transformer_0.0.8.pth', map_location=model_device)
seq2seq = Seq2SeqTransformer(d_model=model_params['MODEL_DIM'], 
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

model.load_state_dict(checkpoint['model_state_dict'])

# Load pretrained reward model - 
reward_model = LinearRewardModel(input_size=80)

#TODO: load weights for reward model 

#TODO: load list of prompts for rlhf 

#TODO: pass it to RLHF trainer to train

