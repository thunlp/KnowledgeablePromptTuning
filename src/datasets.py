import torch
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import Dataset
import logger
import csv
from typing import Dict, List, Optional, Union, Callable, Tuple
from transformers import GPT2Tokenizer
log = logger.get_logger(__name__)


class DictDataset(Dataset):
    """A dataset of tensors that uses a dictionary for key-value mappings"""

    def __init__(self, **tensors):
        tensors.values()
        assert all(next(iter(tensors.values())).size(0) == tensor.size(0) for tensor in tensors.values())
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)

    def cuda(self):
        for key in self.tensors:
            self.tensors[key] = self.tensors[key].cuda()

    def cpu(self):
        for key in self.tensors:
            self.tensors[key] = self.tensors[key].cpu()


class TCDataset(DictDataset):
    def __init__(self, args, tokenizer, split, shot=0, repetition=-1):
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.repetition = repetition
        self.shot=shot
        pass
    
    def delete_common_words(self, d):
        word_count = {}
        for key in d:
            for w in d[key]:
                if w not in word_count:
                    word_count[w]=1
                else:
                    word_count[w]+=1
        for w in word_count:
            if word_count[w]>=2:
                for key in d:
                    if w in d[key][1:]:
                        findidx = d[key][1:].index(w)
                        d[key].pop(findidx+1)
        return d


    def sample_training_set(self, all_examples, exclude=None):
        if not(self.args.num_examples_per_label>0 or self.args.num_examples_total>0):
            return all_examples

        if exclude is None:
            exclude = []
        selected_ids = []
        if self.args.num_examples_per_label>0:
            labels = [x['label'] for x in all_examples]
            assert max(labels)+1 == len(set(labels))
            ids_per_label = [[] for x in range(max(labels)+1)]
            for idx, l in enumerate(labels):
                if idx not in exclude:
                    ids_per_label[l].append(idx)
            for ids in ids_per_label:
                tmp = np.array(ids)
                np.random.shuffle(tmp)
                selected_ids.extend(tmp[:self.args.num_examples_per_label].tolist())
            selected_ids = np.array(selected_ids)
            np.random.shuffle(selected_ids)
            log.info("Selected examples (equal num for each class) {}".format(selected_ids.tolist()))
        elif self.args.num_examples_total>0:
            all_ids = np.array([i for i in range(len(all_examples)) if i not in exclude])
            np.random.shuffle(all_ids)
            selected_ids = all_ids[:self.args.num_examples_total]
            log.info("Selected examples (mixed) {}".format(selected_ids.tolist()))
        self.selected_ids = selected_ids
        selected_examples = [all_examples[idx] for idx in selected_ids]
        return selected_examples



    def get_labels(self):
        self.label_id_2_name = {}
        with open(os.path.join(self.args.data_dir, self.dirname,"{}.txt".format(self.args.label_word_file)),'r') as fin:
            L = fin.readlines()
            for idx, line in enumerate(L):
                category_words = line.strip().replace(","," ").split()
                self.label_id_2_name[idx] = category_words#[:self.args.label_num]
        self.label_id_2_name=self.delete_common_words(self.label_id_2_name)
        prompt_label_idx = []
        for i in range(len(self.label_id_2_name)):
            ws_ids = []
            for w in self.label_id_2_name[i]:
                ws = [ w ]
                for w_new in ws:
                    if not self.temps["mask_first"]:
                        w_new = " "+w_new
                    # print(w_new)

                    w_ids = self.tokenizer.encode(w_new, add_special_tokens=False)
                    if self.args.multitoken==0 and len(w_ids)>1:
                        continue
                    ws_ids.append(w_ids)
                    

                    # print(ws_ids)
            prompt_label_idx.append(ws_ids)
        # print(prompt_label_idx)
        
        
        shortest = min([len(i) for i in prompt_label_idx])
        assert shortest>0, "shortest label set has no words"
        # cutoff = min([shortest, self.args.label_num])  ### 0628change
        # prompt_label_idx = [i[:cutoff] for i in prompt_label_idx] ### 0628change
        # log.info("Prompt label idx {}".format([self.tokenizer.convert_ids_to_tokens(i) for i in prompt_label_idx]))
        # np.save("prompt_label_idx_dbpedia.npy", np.array(prompt_label_idx))
        # exit()

        

        self.prompt_label_idx = prompt_label_idx
        # log.debug("prompt_label_idx {}".format(self.prompt_label_idx))
        # return prompt_label_idx
        # self.prompt_label_idx = torch.tensor(prompt_label_idx)
        # log.debug("prompt_label_idx size {}".format(self.prompt_label_idx.size()))
        


    @staticmethod
    def shortenable(s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True

    def get_mask_positions(self, input_ids):
        labels = [-1] * len(input_ids)
        if not self.args.finetuning:
            label_idx = input_ids.index(self.mask_id)
            labels[label_idx] = 1
        return labels

    def list2tensor(self, data):
        res = {}
        res['input_ids'] = []
        res['token_type_ids'] = []
        res['attention_mask'] = []
        res['input_flags'] = []
        res['mlm_labels'] = []
        res['labels'] = []
        res['entity_kgembs'] = []
        res['converted_pos'] = []
        res['label_word_ids'] = []

        
        for idx, i in enumerate(tqdm(data)):
            input_ids, token_type_ids = self.tokenize(i)
            attention_mask = [1] * len(input_ids)
            padding_length = self.args.max_seq_length - len(input_ids)
            
            attention_mask = [1] * len(input_ids)
            padding_length = self.args.max_seq_length - len(input_ids)

            if padding_length < 0:
                raise ValueError(f"Maximum sequence length is too small, got {len(input_ids)} input ids")

            input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)

            assert len(input_ids) == self.args.max_seq_length
            assert len(attention_mask) == self.args.max_seq_length
            assert len(token_type_ids) == self.args.max_seq_length
            mlm_labels = self.get_mask_positions(input_ids)
            res['input_ids'].append(input_ids)
            res['mlm_labels'].append(mlm_labels)
            res['attention_mask'].append(attention_mask)
            res['token_type_ids'].append(token_type_ids)
            res['labels'].append(i['label'])

        tensor_res = {}
        for key in res:
            if len(res[key])>0:
                tensor_res[key] = torch.Tensor(res[key]).long()

        log.info("label size{}".format(tensor_res['labels'].size()))
        return tensor_res

    def get_parts(self, part_a, part_b):
        text_a = self.shortenable(part_a)
        text_b = self.shortenable(part_b)

        composed = [[],[]]
        cur=0
        for x in self.temps['text']:
            if x=="<mask>":
                composed[cur].append(self.tokenizer.mask_token)
            elif x=="<Mask>":
                composed[cur].append(self.tokenizer.mask_token)
            elif x=="<a>":
                composed[cur].append(text_a)
            elif x=="<b>":
                composed[cur].append(text_b)
            elif x=="<a!.>":
                text_tmp = part_a.strip(".")
                text_tmp = self.shortenable(text_tmp)
                composed[cur].append(text_tmp)
            elif x=="<|>":
                cur=1
            else:
                composed[cur].append(x)
        return composed

    @property
    def mask(self) -> str:
        """Return the underlying LM's mask token"""
        return self.tokenizer.mask_token

    @property
    def mask_id(self) -> int:
        """Return the underlying LM's mask id"""
        return self.tokenizer.mask_token_id

    def tokenize(self, data):
        # log.debug("xxx {}".format(data))
        # exit()
        parts_a = data['text_a']
        parts_b = data['text_b']
        label = data['label']
        parts_a, parts_b = self.get_parts(parts_a, parts_b)
        
        kwargs = {'add_prefix_space': True} if isinstance(self.tokenizer, GPT2Tokenizer) else {}

        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        parts_a = [(self.tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_a if x]

        # log.debug("xxxx {}".format(parts_a))
        # exit()
        parts_a_c = parts_a.copy()
        parts_b_c = parts_b.copy()
        if parts_b:
            parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
            parts_b = [(self.tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_b if x]

        try:
            self.truncate(parts_a, parts_b, max_length=self.args.max_seq_length)
        except:
            log.error("{} |||  {}  |||| {} |||| {} ||| {}".format(data['text_a'], data['text_b'], parts_a_c, parts_b_c, parts_a))

        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else None

        # log.debug("==={}===".format(self.tokenizer.convert_ids_to_tokens(tokens_a)))
        # exit()
        input_ids = self.tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

        return input_ids, token_type_ids
    
    def get_template(self, template_id=0):
        temps = {}
        if self.args.finetuning:
            template =['<a>', '<b>']
        else:
            
            template_file = open(os.path.join(self.args.data_dir,self.dirname, self.args.template_file_name),'r') 
            templates = [line.strip().split() for line in template_file]
            template_id = self.args.template_id
            template = templates[template_id]
        temps['text'] = template
        self.temps = temps
        ## get whether the verbalize is the begining or the middle.
        if "<Mask>" in template:
            self.temps['mask_capitalize'] = True
            if template.index("<Mask>")==0:
                self.temps["mask_first"] = True
            else:
                self.temps["mask_first"] = False
        elif "<mask>" in template:
            self.temps['mask_capitalize'] = False
            if template.index("<mask>")==0:
                self.temps["mask_first"] = True
            else:
                self.temps["mask_first"] = False
        else:
            self.temps["mask_first"] = False
            self.temps['mask_capitalize'] = False

        
        log.debug(self.temps)

    def truncate(self, parts_a: List[Tuple[str, bool]], parts_b: List[Tuple[str, bool]], max_length: int):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        total_len += self.tokenizer.num_special_tokens_to_add(bool(parts_b))
        num_tokens_to_remove = total_len - max_length

        
        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)

    @staticmethod
    def _remove_last(parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])
    
    @staticmethod
    def _seq_length(parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    def save(self):
        pass


# class AgnewsDataset(TCDataset):
#     def __init__(self, args, tokenizer, split, shot=0, repetition=-1):
#         super(AgnewsDataset, self).__init__(args, tokenizer, split,shot, repetition)
#         self.dirname = "agnews"
#         self.get_labels()
#         self.get_template()
#         self.get_tensors(split = split, shot=shot, repetition=repetition)
    
#     def get_tensors(self,split="train", shot=0, repetition=0):
#         if split=='test' or shot==0:
#             features = []
#             label_file  = open(os.path.join(self.args.data_dir,self.args.dataset,"{}_labels.txt".format(split)),'r') 
#             labels  = [int(x.strip()) for x in label_file.readlines()]
#             with open(os.path.join(self.args.data_dir,self.args.dataset,"{}.txt".format(split)),'r') as fin:
#                 for idx, line in enumerate(fin):
#                     line = line.strip()
#                     features.append({"text_a":line, "text_b":"", "label":labels[idx]})
#             self.tensors = self.list2tensor(features)
#         else:
#             log.info("loading split {} shot {}".format(split,shot))
#             features = []
#             # label_file  = open(os.path.join(self.args.data_dir,self.args.dataset,"{}_labels.txt".format(split)),'r') 
#             # labels  = [int(x.strip()) for x in label_file.readlines()]
#             with open(os.path.join(self.args.data_dir,self.args.dataset,'{}_{}-shot_rep-{}.json'.format(split, shot, repetition)),'r') as fin:
#                 for idx, line in enumerate(fin):
#                     line = eval(line.strip())
#                     line['text_a'] = line['text']
#                     line['text_b'] = ""
#                     features.append(line)
#             self.tensors = self.list2tensor(features)
        
class AgnewsDataset(TCDataset):
    num_class=4
    def __init__(self, args, tokenizer, split, shot=0, repetition=-1, exclude=None):
        super(AgnewsDataset, self).__init__(args, tokenizer, split,shot, repetition)
        self.exclude_ids= exclude
        self.dirname = "agnews"
        self.get_template()
        self.get_labels()
        self.get_tensors_from_csv(split = split)
    
    
    def get_tensors_from_csv(self, split='train'):
        
        path = os.path.join(self.args.data_dir,self.dirname,"{}.csv".format(split))
        features = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')
                features.append({"text_a": text_a, "text_b":text_b, "label":int(label)-1})
            if split in ['train', 'valid']:
                features = self.sample_training_set(features, exclude=self.exclude_ids)
            self.tensors = self.list2tensor(features)
    
    @staticmethod
    def get_test_labels_only(data_dir, dirname):
        path = os.path.join(data_dir, dirname,"{}.csv".format("test"))
        labels = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                labels.append(int(label)-1)
        return labels

    

   
class DBpediaDataset(TCDataset):
    num_class=14
    def __init__(self, args, tokenizer, split, shot=0, repetition=-1, exclude=None):
        super(DBpediaDataset, self).__init__(args, tokenizer, split,shot, repetition)
        self.dirname = "dbpedia"
        self.exclude_ids= exclude
        self.get_template()
        self.get_labels()
        self.get_tensors_from_txt(split = split)
    
    def get_tensors_from_txt(self,split="train"):
        features = []
        label_file  = open(os.path.join(self.args.data_dir,self.dirname,"{}_labels.txt".format(split)),'r') 
        labels  = [int(x.strip()) for x in label_file.readlines()]
        with open(os.path.join(self.args.data_dir,self.dirname,'{}.txt'.format(split)),'r') as fin:
            for idx, line in enumerate(fin):
                splited = line.strip().split(". ")
                # log.debug(splited)
                # exit()
                text_a, text_b = splited[0], splited[1:]
                text_a = text_a+"."
                text_b = ". ".join(text_b)
                features.append({"text_a": text_a, "text_b":text_b, "label":int(labels[idx])})
        if split in ['train', 'valid']:
            features = self.sample_training_set(features, exclude=self.exclude_ids)
        self.tensors = self.list2tensor(features)
    
    @staticmethod
    def get_test_labels_only(data_dir, dirname):
        label_file  = open(os.path.join(data_dir,dirname,"{}_labels.txt".format('test')),'r') 
        labels  = [int(x.strip()) for x in label_file.readlines()]
        return labels
    
  
   
class AmazonDataset(TCDataset):
    num_class=2
    def __init__(self, args, tokenizer, split, shot=0, repetition=-1, exclude=None):
        super(AmazonDataset, self).__init__(args, tokenizer, split,shot, repetition)
        self.dirname = "amazon"
        self.exclude_ids= exclude
        self.get_template()
        self.get_labels()
        self.get_tensors_from_txt(split = split)
    
    def get_tensors_from_txt(self,split="train"):
        features = []
        label_file  = open(os.path.join(self.args.data_dir,self.dirname,"{}_labels.txt".format(split)),'r') 
        labels  = [int(x.strip()) for x in label_file.readlines()]
        if split=="test":
            log.info("Sample a mid-size test set for effeciecy, use sampled_test_idx.txt")
            with open(os.path.join(self.args.data_dir,self.dirname,"sampled_test_idx.txt"),'r') as sampleidxfile:
                sampled_idx = sampleidxfile.readline()
                sampled_idx = sampled_idx.split()
                sampled_idx = set([int(x) for x in sampled_idx])

        with open(os.path.join(self.args.data_dir,self.dirname,'{}.txt'.format(split)),'r') as fin:
            for idx, line in enumerate(fin):
                if split=='test':
                    if idx not in sampled_idx:
                        continue
                text_a = line.strip()
                text_b = ""
                features.append({"text_a": text_a, "text_b":text_b, "label":int(labels[idx])})
        if split in ['train', 'valid']:
            features = self.sample_training_set(features, exclude=self.exclude_ids)
        self.tensors = self.list2tensor(features)
    
    @staticmethod
    def get_test_labels_only(data_dir, dirname):
        label_file  = open(os.path.join(data_dir,dirname,"{}_labels.txt".format('test')),'r') 
        labels  = [int(x.strip()) for x in label_file.readlines()]
        return labels

    def sample_test_set(self, all_examples):
        
        all_ids = np.array([i for i in range(len(all_examples))])
        np.random.shuffle(all_ids)
        selected_ids = all_ids[:num]
        log.info("Sample {} test samples for efficiency".format(num))
        self.selected_ids = selected_ids
        selected_examples = [all_examples[idx] for idx in selected_ids]
        return selected_examples

        
class ImdbDataset(TCDataset):
    num_class=2
    def __init__(self, args, tokenizer, split, shot=0, repetition=-1, exclude=None):
        super(ImdbDataset, self).__init__(args, tokenizer, split,shot, repetition)
        self.dirname = "imdb"
        self.exclude_ids= exclude
        self.get_template()
        self.get_labels()
        self.get_tensors_from_txt(split = split)
    
    def get_tensors_from_txt(self,split="train"):
        features = []
        label_file  = open(os.path.join(self.args.data_dir,self.dirname,"{}_labels.txt".format(split)),'r') 
        labels  = [int(x.strip()) for x in label_file.readlines()]
        with open(os.path.join(self.args.data_dir,self.dirname,'{}.txt'.format(split)),'r') as fin:
            for idx, line in enumerate(fin):
                text_a = line.strip()
                text_b = ""
                features.append({"text_a": text_a, "text_b":text_b, "label":int(labels[idx])})
        if split in ['train', 'valid']:
            features = self.sample_training_set(features, exclude=self.exclude_ids)
         
        self.tensors = self.list2tensor(features)
    
    @staticmethod
    def get_test_labels_only(data_dir, dirname):
        label_file  = open(os.path.join(data_dir,dirname,"{}_labels.txt".format('test')),'r') 
        labels  = [int(x.strip()) for x in label_file.readlines()]
        return labels

class YelpDataset(TCDataset):
    def __init__(self, args, tokenizer, split, shot=0, repetition=-1, exclude=None):
        super(YelpDataset, self).__init__(args, tokenizer, split,shot, repetition)
        self.dirname = "yelp"
        self.get_template()
        self.get_labels()
        self.get_tensors_from_csv(split = split)
    
    def get_tensors_from_csv(self, split='train'):
        path = os.path.join(self.args.data_dir,self.dirname,"{}.csv".format(split))
        features = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, body = row
                # guid = "%s-%s" % (set_type, idx)
                text_a = body.replace('\\n', ' ').replace('\\', ' ')
                features.append({"text_a": text_a, "text_b":"", "label":int(label)-1})
        if split in ['train', 'valid']:
            features = self.sample_training_set(features, exclude=self.exclude_ids)
        self.tensors = self.list2tensor(features)
    
    @staticmethod
    def get_test_labels_only(data_dir, dirname):
        path = os.path.join(data_dir, dirname,"{}.csv".format("test"))
        labels = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, body = row
                labels.append(int(label)-1)
        return labels

class YahooDataset(TCDataset):
    def __init__(self, args, tokenizer, split, shot=0, repetition=-1, exclude=None):
        super(YahooDataset, self).__init__(args, tokenizer, split,shot, repetition)
        self.dirname = "yahoo"
        self.get_template()
        self.get_labels()
        self.get_tensors_from_csv(split = split)
    
    def get_tensors_from_csv(self, split='train'):
        path = os.path.join(self.args.data_dir,self.dirname,"{}.csv".format(split))
        features = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, question_title, question_body, answer = row
                # guid = "%s-%s" % (set_type, idx)
                text_a = ' '.join([question_title.replace('\\n', ' ').replace('\\', ' '),
                                   question_body.replace('\\n', ' ').replace('\\', ' ')])
                text_b = answer.replace('\\n', ' ').replace('\\', ' ')
                features.append({"text_a": text_a, "text_b":text_b, "label":int(label)-1})
        if split in ['train', 'valid']:
            features = self.sample_training_set(features, exclude=self.exclude_ids)
        self.tensors = self.list2tensor(features)
    
    @staticmethod
    def get_test_labels_only(data_dir, dirname):
        path = os.path.join(data_dir, dirname,"{}.csv".format("test"))
        labels = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, question_title, question_body, answer = row
                labels.append(int(label)-1)
        return labels



Datasets = {
    "agnews": AgnewsDataset,
    "dbpedia": DBpediaDataset,
    "amazon" : AmazonDataset,
    "imdb": ImdbDataset,
    "yelp": YelpDataset,
    "yahoo": YahooDataset
}
