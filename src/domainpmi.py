import logging
import logger
import os
from configs import get_args_parser, get_model_classes, get_args, save_args
args = get_args_parser()
log = logger.setup_applevel_logger(file_name = args.logger_file_name)
# log a few args
log.info("unicode: {}".format(args.random_code))
log.info("ngpu: {}".format(args.n_gpu))
save_args()

import torch.nn as nn
from tqdm import tqdm
from datasets import Datasets
from optimizer import get_optimizer
import random
import torch
import numpy  as np
from model import get_model
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from sklearn.metrics import f1_score, accuracy_score
from criterion import MultiClassCriterion, evaluate
from remove_verbalizer import Removewords


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def get_tokenizer(special=[]):
    args = get_args()
    model_classes = get_model_classes()
    model_config = model_classes[args.model_type]
    tokenizer = model_config['tokenizer'].from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer.add_tokens(special)
    return tokenizer






set_seed(args.seed)
tokenizer = get_tokenizer(special=[])


datasetclass = Datasets[args.dataset]
test_dataset = datasetclass(args=args, tokenizer=tokenizer, split="test")
test_dataset.cuda()
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=50)

repetition=0
total_repetition=5
ave_test = 0





    


while(repetition<total_repetition):
    log.info("begin experiment repetition {}".format(repetition))
    if args.tuning==1:
        train_dataset = datasetclass(args=args, tokenizer=tokenizer, split="train", shot=args.shot, repetition=repetition)
        train_dataset.cuda()
  
        if args.use_train_as_valid:
            val_dataset = train_dataset
        else:
            val_dataset = None
        train_batch_size = args.per_gpu_train_batch_size*args.n_gpu
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
        val_sampler = SequentialSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=50)

    model = get_model(tokenizer, test_dataset.prompt_label_idx)
    if args.tuning:
        optimizer, scheduler, optimizer_new_token, scheduler_new_token = get_optimizer(model, train_dataloader)
        criterion = MultiClassCriterion(args=args)

    epochs=0
    mx_f1 = 0


    tensors_template_only = test_dataset.list2tensor([{'text_a':"",'text_b':"","label":0}])
    logit = model(tensors_template_only)

    


    while (args.tuning  and epochs < args.max_epochs):
        # model.train()
        # model.zero_grad()
        tr_loss = 0.0
        global_step = 0 
        epochs +=1 
        word_remover = Removewords(train_dataset.prompt_label_idx.cpu().tolist())
        with tqdm(total = len(train_dataloader)) as t:
            for step, batch in enumerate(train_dataloader):
                t.set_description("Epoch {}".format(epochs))
                logits,_,att = model(**batch)
                labels = batch['labels']
                # log.debug(labels)
                # log.debug(logits)
                loss,_ = criterion(logits, labels, att)
                if args.dropwords == 1:
                    word_remover.append(logits)
                t.set_postfix(loss=loss.item())
                t.update(1)
            

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer_new_token.step()
                    scheduler_new_token.step()
                    model.zero_grad()
                    # log.debug (args)
                    global_step += 1
                    # log.debug (tr_loss/global_step, mx_f1)
            
            
            
            f1, auc_score = evaluate(model, val_dataloader, args, tokenizer=None, prompt_label_idx=train_dataset.prompt_label_idx)
            if f1 > mx_f1:
                mx_f1 = f1
                torch.save(model.state_dict(), args.output_dir+"/"+'best_model_rep{}'.format(repetition)+".pt")
            
            log.info ("{} {}".format(tr_loss/global_step, f1, mx_f1))
            # break

        if args.dropwords == 1:
            new_prompt_label_idx = word_remover.remove_uninformative()
            train_dataset.prompt_label_idx = new_prompt_label_idx
            val_dataset.prompt_label_idx = new_prompt_label_idx
            test_dataset.prompt_label_idx = new_prompt_label_idx
            model.prompt_label_idx = new_prompt_label_idx

        

        
    log.info("begin testing")
    if args.tuning:
        model.load_state_dict(torch.load(args.output_dir+"/"+'best_model_rep{}'.format(repetition)+".pt"))
    mic, mac= evaluate(model, test_dataloader, args, tokenizer, prompt_label_idx=test_dataset.prompt_label_idx)
    log.info("test f1 {}, {}".format(mic, mac))

    ave_test += mic
    repetition += 1

    if args.tuning:
        del model, train_dataset,train_dataloader,val_dataset, val_dataloader
        torch.cuda.empty_cache()

log.info("ave test f1 {}".format(ave_test/total_repetition))
