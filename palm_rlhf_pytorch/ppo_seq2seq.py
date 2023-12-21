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
from palm_rlhf_pytorch.seq2seq.reward_model import LinearRewardModel

from accelerate import Accelerator

PPOActionCriticReturn = namedtuple('PPOActionCriticReturn', [
    'actions',
    'sequence',
    'mask',
    'prompt_mask',
    'action_logits',
    'values'
])

@beartype
class ActorCritic(nn.Module):
    def __init__(
        self, 
        mto_transformer: Seq2SeqTransformer,
        mto_transformer_critic: Optional,
        pooled_values: bool = False, 
        actor_dropout = 0.,
        critic_dropout = 0.
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
    def generate(self, 
                 orders, 
                 results, 
                 pat_cov, 
                 max_seq_len,
                 eos_token=None, 
                 return_values=False, 
                 **kwargs): 
        '''
        Generate actions using seq2seq model by autoregressively 
        generating tokens
        '''
        actions = self.actor.greedy_decode(
            orders, 
            results, 
            pat_cov, 
            max_seq_len, 
            BOS_idx, 
            EOS_idx
        )

        initial_context = torch.cat([orders, results, pat_cov], dim=-1) #TODO: adjust dimension as needed 
        # Concatenate the initial context with the generated actions to form the initial sequence 
        sequence = torch.cat([initial_context, actions], dim=-1)
        
        initial_context_len = initial_context.shape[-1] #TODO: not sure if this corresponds to the sequence lenght or what??
        total_len = sequence.shape[-1]

        # create prompt mask - boolean mask where True corresponds to the initial context
        prompt_mask = torch.arange(total_len) < initial_context_len
        prompt_mask = prompt_mask.unsqueeze(0).expand(sequence.shape[0], -1) #Expand across batch size
        action_mask = ~prompt_mask

        # handle optional presence of an EOS token
        mask = None
        if exists(eos_token): 
            mask = ((sequence == eos_token).cumsum(dim = -1) == 0)
            mask = F.pad(mask, (1, -1), value = True) # include eos token
            action_mask &= mask 

        # TODO: pass sequence through forward() to calculate critic values 
        action_logits, value = self.forward(
            sequence, 
            mask = action_mask, 
            return_values = return_values
        )

        return PPOActionCriticReturn(
            actions, 
            sequence, 
            mask, 
            prompt_mask, 
            action_logits, 
            values
        )
        
    def forward(
        self, 
        sequence, #concatenation of state and actions
        mask=None, 
        return_values=True
    ):
        '''forward pass through the actor critic network'''
        #TODO: make sure that sequence dim does not surpass max dimensions
        action_logits = self.actor(
            sequence
        )

        if not return_values: 
            return action_logits, None 
        
        critic_embeds = self.critic_palm(
            sequence, 
            return_only_embedding = True
        )

        if self.pooled_values: 
            critic_embeds = shift(critic_embeds, shift=1, dim=-2)
            critic_embeds = masked_mean(critic_embeds, mask, dim=1)

        values = self.value_head(critic_embeds) #output of critic model
        
        return action_logits, values 
        
@beartype()
class RLHFTrainer(nn.Module):
    def __init__(
        self, 
        *, 
        prompts: Optional[List[str]] = None, 
        prompts_path: Optional[str] = None, 
        prompt_token_ids: Optional[torch.Tensor] = None, 
        tokenizer: Callable = None, 
        mto: Seq2SeqTransformer, 
        rewardmodel: LinearRewardModel,
        critic: Optional[ActorCritic] = None, 
        actor_critic: Optional[ActorCritic],
        actor_lr = 1e-4,
        critic_lr = 1e-4,
        actor_wd = 0.,
        critic_wd = 0.,
        actor_adam_eps = 1e-7,
        critic_adam_eps = 1e-7,
        critic_pooled_values = True,
        actor_dropout = 0.,
        critic_dropout = 0.,
        betas = (0.9, 0.999),
        max_norm = None,
        eps_clip = 0.2,
        value_clip = 0.4,
        beta_s = .01,
        pad_value = 0.,
        minibatch_size = 16,
        epochs = 1,
        kl_div_loss_weight = 0.1, # between old action probs and new action probs - not sure what the right value is
        accelerate_kwargs: dict = {},
        use_lion = False
    ): 
        '''
        trainer for RLHF setup. Combines a pretrained language model, an actor critic model, and a reward model to 
        perform RL training with human feedback 
        '''
        super().__init__()

        # take care of prompts - only one of these should exist
        assert (exists(prompts) + exists(prompts_path) + exists(prompt_token_ids)) == 1

        if exists(prompts_path):
            path = Path(prompts_path)
            prompts = path.read_text().split('/')

        if exists(prompts_path):
            path = Path(prompts_path)
            prompts = path.read_text().split('\n')

        # TODO: Modify here, since I don't have access to the data, perhaphs I do not need a tokenizer, and I can just pass in embeddings
        if exists(prompts):
            assert len(prompts) > 0, 'no prompts'
            assert exists(tokenizer), 'tokenizer must be passed in if raw text prompts are given'

        self.pad_value = pad_value
        self.num_prompts = prompt_token_ids.shape[0]
        self.register_buffer('prompt_token_ids', prompt_token_ids) #save to state dict without applying gradients during training

        # initialize models 

        self.mto = mto #multi temporal order transformer

        if not exists(actor_critic):
            actor_critic = ActorCritic(
                mto_transformer=mto, 
                mto_transformer_critic=mto, 
                pooled_values=False, 
                actor_dropout=actor_dropout, 
                critic_dropout=critic_dropout
            )

        self.actor_critic = actor_critic

        self.reward_model = reward_model.eval()

        #train hyper parameters 

        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.max_norm = max_norm
        self.kl_div_loss_weight = kl_div_loss_weight

        # optimizers (different for both actor and critic)

        self.actor_optim = get_optimizer(
            actor_critic.actor_parameters(),
            lr=actor_lr, 
            wd=actor_wd, #weight decay 
            betas=betas,
            eps=actor_adam_eps,
            use_lion=use_lion
            )

        self.critic_optim = get_optimizer(
            actor_critic.critic_parameters(),
            lr=critic_lr,
            wd=critic_wd, 
            betas=betas, 
            eps=critic_adam_eps,
            use_lion=use_lion
            )

        # ppo hyperparams 

        self.eps_clip = eps_clip
        self.value_clip = value_clip
        self.beta_s = beta_s

        # prepare with accelerator
        (
            self.actor_critic,
            self.reward_model,
            self.actor_optim,
            self.critic_optim
        ) = self.accelerate.prepare(
            self.actor_critic,
            self.reward_model,
            self.actor_optim,
            self.critic_optim
        )

        def print(self, msg):
            return self.accelerate.print(msg)
        
        def save(self, filepath='./checkpoint.pt'):
            torch.save(self.actor_critic.state_dict(), filepath)
        
        def load(self, filepath='./checkpoint.pt'):
            state_dict = torch.load(file_path)
            self.actor_critic.load_state_dict(state_dict)

        @property
        def device(self):
            return self.accelerate.device

        @torch.no_grad()
        def generate(
            self, 
            max_seq_len, 
            *args, 
            orders, 
            results, 
            pat_cov,
            num_samples = 4, # sample 4 per prompt and select the one with the highest reward 
            **kwargs
        ):
            #TODO: some sort of check here that we have orders, results adn pat_cov to generate responses 







