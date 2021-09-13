import argparse
import torch
import transformers
from transformers import BertConfig, BertTokenizer, BertModel, \
                         RobertaConfig, RobertaTokenizer, RobertaModel, \
                         AlbertTokenizer, AlbertConfig, AlbertModel, RobertaForMaskedLM, BertForMaskedLM, \
                        RobertaForSequenceClassification
import os
import shutil

import logger
import sys
log = logger.get_logger(__name__)

_GLOBAL_ARGS = None

_MODEL_CLASSES = {
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'model':BertForMaskedLM,
    },
    'roberta': {
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'model':RobertaForMaskedLM,
        # 'ftmodel':RobertaForSequenceClassification,
    },
    'albert': {
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        'model':AlbertModel,
    }
}

def get_args_parser():

    parser = argparse.ArgumentParser(description="Command line interface for Relation Extraction.")

    # Required parameters
    parser.add_argument("--rebuild_dataset", default=0, type=int,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--dataset", type=str, )
    parser.add_argument("--model_type", default="albert", type=str, required=True, choices=_MODEL_CLASSES.keys(),
                        help="The type of the pretrained language model to use")
    parser.add_argument("--model_name_or_path", default="albert-xxlarge-v2", type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--output_dir_base", default="outputlogs/", type=str,
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where to store the pre-trained models downloaded from S3.")
    parser.add_argument("--new_tokens", default=5, type=int, 
                        help="The output directory where the model predictions and checkpoints will be written")

    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization for PET. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for PET training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for PET evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass in PET.")
    parser.add_argument("--max_epochs", default=3, type=int)
                        # help="If > 0: set total number of training steps to perform in PET. Override num_train_epochs.")

    # Other optional parameters
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate_for_new_token", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--weight_decay", default=1e-5, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--tuning", default=0, type=int, required='--finetuning' not in sys.argv,
                        help="0 for not tuning the prompt")
    parser.add_argument("--pred_method", default="none", type=str,required='--finetuning' not in sys.argv, help=" method for get preds from logits")
    parser.add_argument("--optimize_method", default="none", type=str, required='--finetuning' not in sys.argv, help=" method for get preds from logits")

    parser.add_argument("--label_num", default=1, type=int, help=" num_labels_to_use")
    # parser.add_argument("--shot", type=int, default=16)
    # parser.add_argument("--repetition", type=int, default=1)

    parser.add_argument("--finetuning", action='store_true', help="finetuning instead of prompt tuning")

    parser.add_argument("--div_prior", type=int, default=-1, required='--finetuning' not in sys.argv,
                        help=" whether to divide the prior of outputing the mask, -1 means automatically determine by zero-shot or few-shot ")
    parser.add_argument("--label_word_file", type=str)
    parser.add_argument("--topk", type=int, default=-1)
    parser.add_argument("--topkratio",type=float, default=-1)
    parser.add_argument("--autotopk",type=int, default=1)

    # parser.add_argument("--topkratioperclass", type=float, default=-1)
    # parser.add_argument("--dropwords", type=int, default=0, help="drop the verbalizer that are least possible for model prediction")
    parser.add_argument("--template_file_name",type=str, default="template.txt")
    parser.add_argument("--template_id", type=int, default=0)
    parser.add_argument("--num_examples_total", type=int, default=-1)
    parser.add_argument("--num_examples_per_label",type=int, default=-1)
    # parser.add_argument('--use_train_as_valid', type=int, default=1)
    # parser.add_argument('--run_to_save_all_logits', type=int, default=0)
    # parser.add_argument('--compute_prior', type=int, default=0)
    parser.add_argument('--cut_off', type=float, default=0.5, help="threshold for dropping the verbalizer that are least possible for model prediction", required='--finetuning' not in sys.argv)
    parser.add_argument('--task', type = str, default='run_single_template', required=True)
    parser.add_argument("--only_tune_last_layer", type=int, default=0)
    parser.add_argument("--weight_learning_rate", type=float, default=0.01)
    parser.add_argument("--pred_temperature", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=2 )
    parser.add_argument("--multitoken", type=int, default=0, required='--finetuning' not in sys.argv,)
    parser.add_argument("--tf_idf_filtering", type=int, default=0,)
    parser.add_argument("--filtering_using_prior", type=int ,default=0, required='--finetuning' not in sys.argv,)

    parser.add_argument("--learning_rate_for_classifier", type=float, default=1e-2, required='--finetuning' in sys.argv,)
    parser.add_argument("--domainpmi", type=int, default=0)

    

    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()

    # if args.div_prior==-1:
    #     args.div_prior= 1-args.tuning
        
    ## check arguments:
    assert not(args.num_examples_total>0 and args.num_examples_per_label>0), "can not set these two argument at the same time"
    ##

    if not os.path.exists(args.output_dir_base):
        os.mkdir(args.output_dir_base)

    change_output_dir(args)

    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args

    return args

def get_args():
    return _GLOBAL_ARGS

def get_model_classes():
    return _MODEL_CLASSES

def save_args():
    args = _GLOBAL_ARGS
    dic = args.__dict__
    with open(os.path.join(args.output_dir,"configs.txt"), 'w') as f:
        for key in dic:
            f.write("{}\t{}\n".format(key,dic[key]))
    log.info("config saved!")

def get_next_global_code(output_dir):
    filelist = os.listdir(output_dir)
    filecode = []
    for file in filelist:
        try:
            x = int(file)
        except ValueError:
            continue
        filecode.append(x)
    if len(filecode)>0:
        cur_code = max(filecode)+1
    else:
        cur_code = 0
    cur_code_str = str(cur_code).rjust(5,'0')
    return cur_code_str

def change_output_dir(args):
    random_code = get_next_global_code(args.output_dir_base)
    args.random_code = str(random_code)
    args.output_dir = os.path.join(args.output_dir_base, random_code)
    os.mkdir(args.output_dir)
    args.logger_file_name = os.path.join(args.output_dir,"output.log")
    



