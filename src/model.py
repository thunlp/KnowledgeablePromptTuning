import torch
import torch.nn as nn
from configs import get_model_classes, get_args
import numpy as np
import logger
log = logger.get_logger(__name__)

class Model(torch.nn.Module):
    def __init__(self, args, tokenizer = None, prompt_label_idx = None):
        log.debug("use prompt tuning model!")
        super().__init__()
        model_classes = get_model_classes()
        model_config = model_classes[args.model_type]
        self.args = args
        self.prompt_label_idx = prompt_label_idx
        # self.prompt_label_idx = torch.arange(self.model.embeddings.word_embedding.size(0))


        # log.debug([tokenizer.convert_ids_to_tokens(prompt_label_idx[i].tolist()) for i in range(4)])
        # log.debug(prompt_label_idx)
        # exit()

        self.tokenizer = tokenizer
        
        self.model = model_config['model'].from_pretrained(
            args.model_name_or_path,
            return_dict=False,
            cache_dir=args.cache_dir if args.cache_dir else None)
        
        if args.only_tune_last_layer==1:
            # log.debug("models's named parameters {}".format([key for key in self.model.named_parameters()]))
            self.not_tunable_parameters = [n for n,p in self.model.named_parameters() if "lm_head" not in n]
            # log.info("not tunable parameters {}".format(self.not_tunable_parameters))
        else:
            self.not_tunable_parameters = []
        log.info("num of not tunable parameters {}".format(len(self.not_tunable_parameters)))

        # self.mlp = torch.nn.Sequential(
        #     torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size))

        # self.extra_token_embeddings = nn.Embedding(args.new_tokens, self.model.config.hidden_size)

    def forward(self, input_ids, attention_mask, token_type_ids, mlm_labels, labels):
        cur_batchsize = input_ids.size(0)
        # log.debug("input_ids {}".format(input_ids[0].cpu().tolist()))
        # exit()
        raw_embeddings = self.model.roberta.embeddings.word_embeddings(input_ids)
        inputs_embeds = raw_embeddings
    
        logits = self.model(inputs_embeds=inputs_embeds,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)[0]
        logits_all = logits[mlm_labels > 0].view(cur_batchsize, 1, -1)
        logits_all = logits_all[:, 0, :]
        return  logits_all #, attention_weight


class FineTuningModel(torch.nn.Module):
    def __init__(self, args):
        log.debug("use fine tuning model!")
        super().__init__()
        model_classes = get_model_classes()
        model_config = model_classes[args.model_type]
        self.args = args
       
        
        self.model = model_config['ftmodel'].from_pretrained(
            args.model_name_or_path, num_labels=args.num_class)
        # self.mlp = torch.nn.Linear(self.model.config.hidden_size, args.num_class)
        self.not_tunable_parameters = []
        

    def forward(self, input_ids, attention_mask, token_type_ids, mlm_labels, labels):
        # cur_batchsize = input_ids.size(0)
        # raw_embeddings = self.model.roberta.embeddings.word_embeddings(input_ids)
        # inputs_embeds = raw_embeddings
    
        outputs  = self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids, labels=labels)
        # log.info("output {}".format(outputs))
        loss = outputs[0]
        logits = outputs[1]
        # logits_all = logits[mlm_labels > 0].view(cur_batchsize, 1, -1)
        # logits_all = logits_
        return  loss, logits#, attention_weight


def get_model(tokenizer=None, prompt_label_idx=None):
    args = get_args()
    # if kmodel:
    #     model = KModel(args, tokenizer, prompt_label_idx)
    # else:
    if args.finetuning:
        model = FineTuningModel(args)
    else:
        model = Model(args, tokenizer, prompt_label_idx)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()
    # model.embedding_weight_0 = model.model.embeddings.word_embeddings.weight[model.prompt_label_idx[0]].transpose(1,0)
    # model.embedding_weight_1 = model.model.embeddings.word_embeddings.weight[model.prompt_label_idx[1]].transpose(1,0)
    # model.embedding_weight_2 = model.model.embeddings.word_embeddings.weight[model.prompt_label_idx[2]].transpose(1,0)
    return model
