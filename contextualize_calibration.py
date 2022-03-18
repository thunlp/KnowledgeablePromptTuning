
from yacs.config import CfgNode
from openprompt.data_utils import FewShotSampler
from torch.utils.data.dataset import Dataset
from transformers.data.processors.utils import InputExample
from openprompt.pipeline_base import PromptDataLoader, PromptModel, PromptForClassification
from typing import *
import torch
# from openprompt.utils.custom_tqdm import tqdm
from tqdm import tqdm



def calibrate(prompt_model: PromptForClassification, dataloader: PromptDataLoader) -> torch.Tensor:
    r"""Calibrate. See `Paper <https://arxiv.org/abs/2108.02035>`_
    
    Args:
        prompt_model (:obj:`PromptForClassification`): the PromptForClassification model.
        dataloader (:obj:`List`): the dataloader to conduct the calibrate, could be a virtual one, i.e. contain an only-template example.
    
    Return:
        (:obj:`torch.Tensor`) A tensor of shape  (vocabsize) or (mask_num, vocabsize), the logits calculated for each word in the vocabulary
    """
    all_logits = []
    prompt_model.eval()
    for batch in tqdm(dataloader,desc='ContextCali'):
        batch = batch.to(prompt_model.device)
        logits = prompt_model.forward_without_verbalize(batch)
        all_logits.append(logits.detach())
    all_logits = torch.cat(all_logits, dim=0)
    return all_logits

