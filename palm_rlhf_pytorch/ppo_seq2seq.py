import math
from pathlib import Path
import copy
from tqdm import tqdm
from functools import partial
from collections import deque, namedtuple
from random import randrange

from beartype import beartype
from beartype.typing import List, Optional, Callable, Deque

import torch
from torch import nnpalm
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from palm_rlhf_pytorch.palm import PaLM
from palm_rlhf_pytorch.reward import RewardModel
from palm_rlhf_pytorch.optimizer import get_optimizer
from palm_rlhf_pytorch.utils import masked_mean, eval_decorator

#seq2seq imports 
from palm_rlhf_pytorch.seq2seq.multi_temporal_order_transformer import Seq2SeqTransformer

@beartype
class ActorCritic(nn.Module):
    def __init__(
        self, 
        mto_transformer: Seq2SeqTransformer,
        mto_transformer_critic: Optional,
        # pooled_values: bool = False
    ):
        super().__init__()
        self.actor = mto_transformer

        self.critic = mto_transformer_critic

        if not exists(self.critic):
            self.critic = copy.deepcopy(mto_transformer)

        #TODO: figure out what this means still
        self.value_head = nn.Sequential(
            nn.Linear(mto_transformer.d_model, 1), #where mto_transformer.dim = ffn_hid_dim
            Rearrange('... 1 -> ...')
        )

        nn.init.zeros_(self.value_head[0].bias)
        nn.init.orthogonal_(self.value_head[0].weight, gain = math.sqrt(2))

    def actor_parameters(self):
        '''
        Get the parameters of the actor network
        '''
        return self.actor.parameters()
    
    def critic_parameters(self):
        '''
        Get the parameters of the critic network 
        '''
        return [*self.critic.parameters(), *self.value_head.parameters()]

    @torch.no_grad()
    @eval_decorator
    def merge_finetune_params():
        