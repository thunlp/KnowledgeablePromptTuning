import torch
from torch import optim
from configs import get_args
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import logger
log = logger.get_logger(__name__)

def get_optimizer(model, train_dataloader, criterion=None):

    args = get_args()
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs

    cur_model = model.module if hasattr(model, 'module') else model

    

    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in cur_model.model.named_parameters() if (not any(nd in n for nd in no_decay)) and (n not in cur_model.not_tunable_parameters)],'weight_decay': args.weight_decay},
        {'params': [p for n, p in cur_model.model.named_parameters() if (any(nd in n for nd in no_decay)) and (n not in cur_model.not_tunable_parameters)],'weight_decay': 0.0}
    ]

    if criterion is not None:
        optimizer_crit_parameters = [{'params': [p for p in criterion.parameters()]}]
    
        crit_optimizer = Adam(optimizer_crit_parameters, lr = args.weight_learning_rate)
        # log.debug("crit paras {}".format(optimizer_crit_parameters))
    else:
        crit_optimizer = None



    # log.info("optimization information {}, {}".format({'params': [n for n, p in cur_model.model.named_parameters() if (not any(nd in n for nd in no_decay)) and (n not in cur_model.not_tunable_parameters)],'weight_decay': args.weight_decay},
        # {'params': [n for n, p in cur_model.model.named_parameters() if (any(nd in n for nd in no_decay)) and (n not in cur_model.not_tunable_parameters)],'weight_decay': 0.0}))

    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=args.learning_rate, 
        # lr = 0, 
        eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    
    # embedding_parameters = [
    #     {'params': [p for p in cur_model.mlp.parameters()]},
    #     {'params': [p for p in cur_model.extra_token_embeddings.parameters()]}
    # # ]
    # embedding_optimizer = AdamW(
    #     embedding_parameters, 
    #     lr=args.learning_rate_for_new_token, 
    #     eps=args.adam_epsilon)
    # embedding_scheduler = get_linear_schedule_with_warmup(
    #     embedding_optimizer, 
    #     num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    

    return optimizer, scheduler, None, None,  crit_optimizer

def get_optimizer_finetuning(model, train_dataloader):

    args = get_args()
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs

    cur_model = model.module if hasattr(model, 'module') else model

    

    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in cur_model.model.named_parameters() if (not any(nd in n for nd in no_decay)) and ('classifier' not in n) ],'weight_decay': args.weight_decay},
        {'params': [p for n, p in cur_model.model.named_parameters() if (any(nd in n for nd in no_decay)) and ('classifier' not in n)  ],'weight_decay': 0.0}
    ]

    optimizer_grouped_parameters2 = [
        {'params': [p for n, p in cur_model.model.named_parameters() if (not any(nd in n for nd in no_decay)) and ('classifier'  in n) ],'weight_decay': args.weight_decay},
        {'params': [p for n, p in cur_model.model.named_parameters() if (any(nd in n for nd in no_decay)) and ('classifier'  in n)  ],'weight_decay': 0.0}
    ]

    # log.debug("optpara {} {}".format(optimizer_grouped_parameters, optimizer_grouped_parameters2))

    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=args.learning_rate, 
        # lr = 0, 
        eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    
    optimizer_classifier = AdamW(
        optimizer_grouped_parameters2, 
        lr=args.learning_rate_for_classifier, 
        # lr = 0, 
        eps=args.adam_epsilon)
    scheduler_classifier = get_linear_schedule_with_warmup(
        optimizer_classifier, 
        num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    


    return optimizer, scheduler, optimizer_classifier, scheduler_classifier
