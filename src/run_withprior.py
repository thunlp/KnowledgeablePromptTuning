import logging
from math import cos
from operator import add
from re import L, VERBOSE
from numpy.lib.function_base import flip

from torch import optim
from torch.utils import data
import logger
import os
from configs import get_args_parser, get_model_classes, get_args, save_args
args = get_args_parser()
log = logger.setup_applevel_logger(file_name = args.logger_file_name)
# log a few args

log.info("unicode: {} dataset: {} template_id {} seed {} margin {} weightlr {} pred_temperature {} pred_method {}".format(args.random_code, args.dataset,
           args.template_id, args.seed, args.margin, args.weight_learning_rate, args.pred_temperature, args.pred_method ))
save_args()

import torch.nn as nn
from tqdm import tqdm
from datasets import Datasets
from optimizer import get_optimizer, get_optimizer_finetuning
import random
import torch
import numpy  as np
from model import get_model
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, dataloader
from sklearn.metrics import f1_score, accuracy_score
from criterion import Criterion
from sklearn.metrics.pairwise import cosine_similarity

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


class Runner(object):
    def __init__(self, args):
        super(Runner,self).__init__()
        self.args = args
        set_seed(args.seed)
        self.tokenizer = get_tokenizer(special=[])
        self.datasetclass = Datasets[args.dataset]
        

    def __get_testdataset(self):
        test_dataset = self.datasetclass(args=self.args, tokenizer=self.tokenizer, split="test")
        test_dataset.cuda()
        self.test_dataset = test_dataset
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=5)
        self.test_dataloader = test_dataloader

    def __get_traindataset(self):
        train_dataset= self.datasetclass(args=args, tokenizer=self.tokenizer, split="train")
        train_dataset.cuda()
        self.train_dataset = train_dataset
        train_batch_size = self.args.per_gpu_train_batch_size*self.args.n_gpu
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
        self.train_dataloader = train_dataloader
    
    def __get_dominpmi_prior(self):
        features = [{"text_a":"", "text_b":"", "label": 0}]
        tensors = self.train_dataset.list2tensor(features)
        for key in tensors:
            if isinstance(tensors[key], torch.Tensor):
                tensors[key] = tensors[key].cuda()
        
        logit_all = self.model(**tensors)
        self.prior_logits = logit_all
        # log.debug("tensors {}".format(logit_all.size()))
        # exit()
        
        # pass
    
    def __get_validdataset(self):
        valid_dataset = self.datasetclass(args=args, tokenizer=self.tokenizer, split="train", exclude = self.train_dataset.selected_ids)
        valid_dataset.cuda()
        self.valid_dataset = valid_dataset
        valid_batch_size = self.args.per_gpu_train_batch_size*self.args.n_gpu
        valid_sampler = RandomSampler(valid_dataset)
        valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=valid_batch_size)
        self.valid_dataloader = valid_dataloader
    
    # def __reload_trainset_and_prompt_labels(self):
    #     self.__get_traindataset()
    #     self.prompt_label_idx = self.train_dataset.prompt_label_idx
    #     self.__get_model()
    #     self.__get_unlabeled_train_logits(num_unlabeled_instance=100)
    #     self.__get_prior()
    #     self.__dropword_with_prior()
    
    def __get_model(self):
        if not self.args.finetuning:
            self.model = get_model(self.tokenizer, prompt_label_idx=self.prompt_label_idx)
        else:
            self.model = get_model()
            

    # def __get_unlabeled_train_logits(self):
    #     all_train_logits = []
    #     all_labels = []
    #     for step, batch in enumerate(self.train_dataloader):
    #         logit_all = self.model(**batch)
    #         all_labels.extend(batch['labels'])
    #         all_train_logits.append(logit_all.detach().cpu().numpy())
    #     all_labels = np.array(all_labels)
    #     all_train_logits = np.concatenate(all_train_logits, axis=0)
    #     self.all_train_logits = all_train_logits
    #     self.all_train_labels = all_labels

    def __get_prior(self, all_logits_all):
        prior_logits = all_logits_all.mean(dim=0).unsqueeze(0)
        log.info(prior_logits)
        log.info(prior_logits.size())
        # prior_logit_saving_dir = "/".join(args.output_dir.split("/")[:-1])+"/template_output/"+ "prior_of_train_template_{}_dataset_{}.npy".format(args.template_id, args.dataset)
        # np.save(prior_logit_saving_dir, prior_logits)
        self.prior_logits = prior_logits

    def __get_prior_context_free():

        pass

    def __dropword_with_prior(self, verbose=False):
        prior_logits = self.prior_logits.detach().cpu().numpy()
        glb_sort = np.argsort(-prior_logits[0])
        remove_set = set(glb_sort[int(len(glb_sort)*self.args.cut_off):].tolist())
        new_prompt_label_idx = []
        def all_not_in(ids, set):
            for i in ids:
                if i in set:
                    return False
            return True
        for prompt_label_idx_perclass in self.prompt_label_idx:
            tmp = []
            tmp_rm = []
            for ids in prompt_label_idx_perclass:
                if all_not_in(ids, remove_set):
                    tmp.append(ids)
                else:
                    tmp_rm.append(ids)
            new_prompt_label_idx.append(tmp)
            
            if verbose:
                tmp_words = [" ".join(self.tokenizer.convert_ids_to_tokens(i)) for i in tmp]
                tmp_rm_words = [" ".join(self.tokenizer.convert_ids_to_tokens(i)) for i in tmp_rm]
                log.info("removed label words: {}, remain words: {}".format(tmp_rm_words, tmp_words))
        self.prompt_label_idx = new_prompt_label_idx
        log.info("num of prompt_idx per class {}".format([len(x) for x in self.prompt_label_idx]))
        # log.info(" prompt_idx per class {}".format( self.prompt_label_idx))

    def __get_optimizer(self):
        optimizer, scheduler, self.optimizer_new_token, self.scheduler_new_token, self.crit_optimizer = get_optimizer(self.model, self.train_dataloader, self.criterion)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def __get_optimizer_finetuning(self):
        self.optimizer, self.scheduler, self.optimizer_classifier, self.scheduler_classifier = get_optimizer_finetuning(self.model, self.train_dataloader)




    def __get_criterion(self):
        criterion = Criterion[self.args.pred_method](args=args, prompt_label_idx=self.prompt_label_idx)
        self.criterion = criterion

    def __tfidf_filtering2(self, all_logits):
        stdss = []
        mean_std = 0
        count=0
        for prompt_label_per_class in self.prompt_label_idx:
            stds = []
            for label in prompt_label_per_class:
                logit = all_logits[:, torch.tensor(label).cuda()]
                logit = logit.mean(dim=-1)
                std = torch.std(logit)
                stds.append(std)
                mean_std+=std
                count+=1
                # embs.append(logit)
            stdss.append(stds)
        mean_std = mean_std/count
        log.info("mean std {}".format(mean_std))
        remove_sets = []
        for classid, stds in enumerate(stdss):
            remove_set1 = []
            for wordid, std in enumerate(stds):
                if std<mean_std:
                    remove_set1.append(wordid)
                # if np.std(sim_one_word)<0.02 and np.mean(sim_one_word)>ave_score.mean():
                #     remove_set2.append(rowid)
            print([self.tokenizer.convert_ids_to_tokens(self.prompt_label_idx[classid][i]) for i in remove_set1])
            # print( [self.tokenizer.convert_ids_to_tokens(self.prompt_label_idx[id][i]) for i in remove_set2])
            remove_sets.append(set(remove_set1))
        prompt_label_idx_new = [[x for id, x in enumerate(self.prompt_label_idx[class_id]) if id not in remove_sets[class_id]] 
                                        for class_id in range(len(self.prompt_label_idx))]
        self.prompt_label_idx = prompt_label_idx_new
        log.info("after tfidf_filter {}".format([len(x) for x in self.prompt_label_idx]))
    
        pass


    def __tfidf_filtering(self):
        model_embeddings = self.model.model.get_input_embeddings()
        all_embs = []
        for prompt_label_per_class in self.prompt_label_idx:
            embs = []
            for label in prompt_label_per_class:
                emb = model_embeddings(torch.tensor(label).cuda())
                emb = emb.mean(0)
                embs.append(emb)
            embs = torch.vstack(embs)
            # log.debug("{}".format(embs.size()))
            all_embs.append(embs)
        
        # target = [x[0][0] for x in self.prompt_label_idx]
        # log.info("{}".format(self.tokenizer.convert_ids_to_tokens(target)))
        # pront(all_embs) 
        target_emb = torch.vstack([x[0] for x in all_embs]).detach().cpu().numpy()
        # log.info("{}".format(target_emb.shape, all_embs[0].detach().cpu().numpy()))

        sim_score = []
        for emb in all_embs:
            sim = cosine_similarity(target_emb, emb.detach().cpu().numpy()).transpose() #(label_num_per_class, class_num)
            # sim = np.argsort(np.argsort(-sim,axis=0),axis=0)
            sim_score.append(sim)
        
        sim_score_cat = np.concatenate(sim_score, axis=0)
        ave_score = np.mean(sim_score_cat, axis=0)
        std_score = np.std(sim_score_cat, axis=0)
        # print(sim_score[0], sim_score[1])
        # sorted_pos = np.argsort(-sim_score[0][:,0])
        # print([self.tokenizer.convert_ids_to_tokens(self.prompt_label_idx[0][i]) for i in  sorted_pos])
        # exit()

        remove_sets = []
        for id, sim in enumerate(sim_score):
            remove_set1, remove_set2 =[], []
            for rowid, sim_one_word in enumerate(sim):
                if sim_one_word[id]<ave_score[id]-0.5*(std_score[id]):
                    remove_set1.append(rowid)
                if np.std(sim_one_word)<0.02 and np.mean(sim_one_word)>ave_score.mean():
                    remove_set2.append(rowid)
            # print([self.tokenizer.convert_ids_to_tokens(self.prompt_label_idx[id][i]) for i in remove_set1])
            # print( [self.tokenizer.convert_ids_to_tokens(self.prompt_label_idx[id][i]) for i in remove_set2])
            remove_sets.append(set(remove_set1+remove_set2))
        
        prompt_label_idx_new = [[x for id, x in enumerate(self.prompt_label_idx[class_id]) if id not in remove_sets[class_id]] 
                                        for class_id in range(len(self.prompt_label_idx))]
        self.prompt_label_idx = prompt_label_idx_new
        log.info("after tfidf_filter {}".format([len(x) for x in self.prompt_label_idx]))

        
        
        
                
    



    def __model_forward(self, dataloader, dataloadertype='Test'):
        all_labels = []
        all_logits_all = []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(dataloader, desc=dataloadertype)):
                # if step>10:
                #     break
                logit_all = self.model(**batch)
                all_labels.append(batch['labels'])  ##0628change
                # all_logits.append(logit.detach().cpu().numpy())   ###0628change
                all_logits_all.append(logit_all)
        all_labels = torch.cat(all_labels, axis=0)
        all_logits_all = torch.cat(all_logits_all, axis=0)
        return all_labels, all_logits_all
        
    def __model_forward_preds(self, dataloader, dataloadertype='Test'):
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(dataloader, desc=dataloadertype)):
                # if step>10:
                #     break
                logit_all = self.model(**batch)
                all_labels.append(batch['labels'])  ##0628change
                # all_logits.append(logit.detach().cpu().numpy())   ###0628change
                # all_logits_all.append(logit_all)
                
                preds = self.criterion.evaluate_batch(logit_all, self.prior_logits, self.tokenizer)
                all_preds.append(preds)

        all_labels = torch.cat(all_labels, axis=0)
        all_preds = np.concatenate(all_preds)
        mic, mac = self.criterion.f1_score(all_preds, all_labels)
        # all_logits_all = torch.cat(all_logits_all, axis=0)
        return mic, mac

    

    def test_model_zero_shot(self):
        # self.args.num_examples_total
        self.args.num_examples_per_label =-1
        # if self.args.domainpmi==0:
        self.__get_traindataset()
        self.prompt_label_idx = self.train_dataset.prompt_label_idx
        self.__get_model()

        if self.args.filtering_using_prior==1 or self.args.div_prior==1:
            if self.args.domainpmi==1:
                self.__get_dominpmi_prior()
            else:
                _, all_logits_all = self.__model_forward(self.train_dataloader, dataloadertype='Unlabel_Support')
                self.__get_prior(all_logits_all)
            if self.args.filtering_using_prior==1:
                self.__dropword_with_prior()
        elif self.args.div_prior==0:
            self.prior_logits=None
        # if self.args.tf_idf_filtering:
        #     self.__tfidf_filtering()
            # self.__tfidf_filtering2(all_logits_all)

        self.__get_criterion()
        if self.args.pred_method=='rank':
            if self.args.autotopk==1:
                best_k = self.criterion.determine_k_base_on_unlabeled_corpus(all_logits_all, self.prior_logits)
            else:
                best_k =  self.args.topkratio
            if self.args.topkratio>0:
                self.criterion.set_k_for_ranking(topkratio=(best_k+self.args.topkratio)/2)
            elif self.args.topkratio==0:
                self.criterion.set_k_for_ranking(topkratio=(best_k+1.0/self.datasetclass.num_class)/2)
            else:
                self.criterion.set_k_for_ranking(topkratio=(best_k))


        
        self.__get_testdataset()

        # self.load_model_and_criterion()
        # self.__get_testdataset()
        log.info("Testing forward...")

        mic, mac = self.__model_forward_preds(self.test_dataloader)
        # mic, mac= self.criterion.evaluate(all_logits_all,all_prior_logits=self.prior_logits, all_labels=all_labels, tokenizer=self.tokenizer)
        log.info("Testing done! MicroF1 {}, MacroF1{}".format(mic, mac))
        # log.info("Testing forward...")
        # all_labels, all_logits_all = self.__model_forward(self.test_dataloader)
        # mic, mac= self.criterion.evaluate(all_logits_all,all_prior_logits=self.prior_logits, all_labels=all_labels, tokenizer=self.tokenizer)
        # log.info("Test f1 {}, {}".format(mic, mac))
        return mic
    
    def test_model_with_different_k(self, ks, add_auto=False):
        self.__reload_trainset_and_prompt_labels()
        all_labels, all_logits_all = self.__get_all_logit_of_testset()
        if add_auto and self.args.pred_method=='rank':
            topk = self.__determine_k_base_on_unlabeled_corpus()
            ks.append(topk)
        mics = []
        for k in ks:
            self.args.topkratio = k
            mic, mac= evaluate(all_logits_all,all_prior_logits=self.prior_logits, prompt_label_idx=self.prompt_label_idx, all_labels=all_labels, args=self.args )
            log.info("k {} test f1 {}, {}".format(k, mic, mac))
            mics.append(mic)
        return mics
    
    
    def inspect_dropped_words(self):
        self.__get_traindataset()
        self.prompt_label_idx = self.train_dataset.prompt_label_idx
        self.__get_model()
        self.__get_unlabeled_train_logits(num_unlabeled_instance=100)
        self.__get_prior()
        self.__dropword_with_prior(verbose=True)
        

    def __train(self):
        self.trace = []
        epochs=0
        while (epochs < args.max_epochs):
            tr_loss = 0.0
            global_step = 0 
            epochs +=1
            log.debug("length of train_dataloader {}".format(len(self.train_dataloader)))
            self.criterion.train()
            with tqdm(total = len(self.train_dataloader)) as t:
                for step, batch in enumerate(self.train_dataloader):
                    t.set_description("Epoch {}".format(epochs))
                    logits_all = self.model(**batch)
                    labels = batch['labels']
                    loss = self.criterion(logits_all, labels, self.prior_logits)
                    t.set_postfix(loss=loss.item())
                    t.update(1)
                
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    loss.backward()
                    tr_loss += loss.item()
                    
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                        self.optimizer.step()
                        self.scheduler.step()
                        # self.optimizer_new_token.step()
                        # self.scheduler_new_token.step()
                        if epochs>=0:
                            self.crit_optimizer.step()
                            self.criterion.zero_grad()

                        self.model.zero_grad()
                        
                        global_step += 1
            
            self.criterion.eval()
            mic = self.__valid()
            

            ## saving ckpt:
            if len(self.trace)==0 or mic>=max(self.trace):
                self.save_model_and_criterion()
            

            # early stop:
            # if len(self.trace)>=self.args.patience and mic<self.trace[-self.args.patience]:
            #     log.info("early stop at epoch {}".format(epochs))
            #     break
            self.trace.append(mic)

    def __train_finetuning(self):
        self.trace = []
        epochs=0
        while (epochs < args.max_epochs):
            tr_loss = 0.0
            global_step = 0 
            epochs +=1
            log.debug("length of train_dataloader {}".format(len(self.train_dataloader)))

            with tqdm(total = len(self.train_dataloader)) as t:
                for step, batch in enumerate(self.train_dataloader):
                    t.set_description("Epoch {}".format(epochs))
                    loss, _ = self.model(**batch)
        
                    t.set_postfix(loss=loss.item())
                    t.update(1)
                
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    loss.backward()
                    tr_loss += loss.item()
                    
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer_classifier.step()
                        self.scheduler_classifier.step()
                        # self.optimizer_new_token.step()
                        # self.scheduler_new_token.step()
                        # if epochs>=0:
                            # self.crit_optimizer.step()
                            # self.criterion.zero_grad()

                        self.model.zero_grad()
                        
                        global_step += 1
            
            mic = self.__test_finetuning(mydataloader=self.valid_dataloader, dataloadername='Valid')
            

            ## saving ckpt:
            if len(self.trace)==0 or mic>=max(self.trace):
                log.info("saving model and criterion!")
                torch.save(self.model.state_dict(), self.args.output_dir+"/model.ckpt")
            

            # early stop:
            # if len(self.trace)>=self.args.patience and mic<self.trace[-self.args.patience]:
            #     log.info("early stop at epoch {}".format(epochs))
            #     break
            self.trace.append(mic)

    def save_model_and_criterion(self):
        log.info("saving model and criterion!")
        torch.save(self.model.state_dict(), self.args.output_dir+"/model.ckpt")
        torch.save(self.criterion.state_dict(), self.args.output_dir+"/criterion.ckpt")
        # log.debug("{}".format(self.criterion.state_dict()))
    
    def load_model_and_criterion(self):
        log.info("Loading model from ckpt!")
        self.model.load_state_dict(torch.load(self.args.output_dir+"/model.ckpt"))
        self.criterion.load_state_dict(torch.load(self.args.output_dir+"/criterion.ckpt"))
        # log.info("k in criterion: {}".format(self.criterion.topk))
    
    def delete_ckpt(self):
        log.info("Deleting ckpt of {}".format(self.args.output_dir))
        os.remove(self.args.output_dir+"/model.ckpt")
        os.remove(self.args.output_dir+"/criterion.ckpt")

    
    def __valid(self):
        self.criterion.eval()
        all_valid_labels, all_valid_logits_all = self.__model_forward(self.valid_dataloader, dataloadertype='Validation')
        if self.args.pred_method=='rank':
            if self.args.autotopk==1:
                best_k = self.criterion.determine_k(all_valid_logits_all, self.prior_logits, labels=all_valid_labels)
            else:
                best_k =  self.args.topkratio
            if self.args.topkratio>0:
                self.criterion.set_k_for_ranking(topkratio=(best_k+self.args.topkratio)/2)
            else:
                self.criterion.set_k_for_ranking(topkratio=(best_k+1.0/self.datasetclass.num_class)/2)
 # a mixture of prior and autodetermined
        mic_val, mac_val= self.criterion.evaluate(all_valid_logits_all,all_prior_logits=self.prior_logits, all_labels=all_valid_labels, tokenizer=self.tokenizer)
        return mic_val
    
    def __test_finetuning(self, mydataloader, dataloadername):
        # self.criterion.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(mydataloader, desc=dataloadername)):
                # if step>10:
                #     break
                _, logits = self.model(**batch)
                # log.debug("logits size {}".format(logits.size()))
                preds = torch.argmax(logits,dim=-1)
                all_labels.append(batch['labels'])
                # all_logits.append(logit.detach().cpu().numpy())  =
                all_preds.append(preds)
 # a mixture of prior and autodetermined

        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        all_preds = all_preds.cpu().numpy()
        all_labels = all_labels.cpu().numpy()
        mi_f1 = f1_score(all_labels, all_preds, average='micro')
        ma_f1 = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)
        log.info("mi_f1 {}, ma_f1 {}, acc {}".format(mi_f1, ma_f1, acc))
        # mic_val, mac_val= self.criterion.evaluate(all_valid_logits_all,all_prior_logits=self.prior_logits, all_labels=all_valid_labels, tokenizer=self.tokenizer)
        return mi_f1


    def __drop_words(self,):
        with tqdm(total = len(self.train_dataloader)) as t:
            all_logits_all = []
            all_labels = []
            for step, batch in enumerate(self.train_dataloader):
                t.set_description("DROPWords:")
                # log.debug("Batch is {}".format(batch))
                logits_all = self.model(**batch)
                labels = batch['labels']
                all_labels.append(labels)
                all_logits_all.append(logits_all)
            all_logits_all = torch.cat(all_logits_all, dim=0)
            all_labels = torch.cat(all_labels,dim=0)
        pass

   
    def __test_after_train(self):
        log.info("Begin testing!")
        self.criterion.eval()
        self.load_model_and_criterion()
        self.__get_testdataset()
        mic, mac = self.__model_forward_preds(self.test_dataloader)
        # mic, mac= self.criterion.evaluate(all_logits_all,all_prior_logits=self.prior_logits, all_labels=all_labels, tokenizer=self.tokenizer)
        log.info("Testing done! MicroF1 {}, MacroF1{}".format(mic, mac))
        return mic



        
    def train_model(self):
        self.__get_traindataset()
        self.__get_validdataset() #this is also need for mean,none,
        self.prompt_label_idx = self.train_dataset.prompt_label_idx
        log.info("{}".format(self.prompt_label_idx))
        self.__get_model()
        if self.args.filtering_using_prior==1 or self.args.div_prior==1:
            log.info("herer")
            _, all_logits_all = self.__model_forward(self.train_dataloader, dataloadertype='Unlabel_Support')
            self.__get_prior(all_logits_all)
            if self.args.filtering_using_prior==1:
                self.__dropword_with_prior()
        elif self.args.div_prior==0:
            self.prior_logits=None
        if self.args.tf_idf_filtering:
            self.__tfidf_filtering()
        self.prior_logits=None
        self.__get_criterion()
        self.__get_optimizer()
        self.__train()
        mic = self.__test_after_train()
        self.delete_ckpt()
        return mic
    
    def finetuning(self):
        self.args.num_class = self.datasetclass.num_class
        self.__get_traindataset()
        self.__get_validdataset()
        self.__get_model()
        # self.__get_criterion()
        # self.criterion=None
        self.__get_optimizer_finetuning()
        self.__train_finetuning()

        log.info("Loading model from ckpt!")
        self.model.load_state_dict(torch.load(self.args.output_dir+"/model.ckpt"))
        self.__get_testdataset()
        self.__test_finetuning(mydataloader=self.test_dataloader, dataloadername='Test' )
        log.info("Deleting ckpt of {}".format(self.args.output_dir))
        os.remove(self.args.output_dir+"/model.ckpt")


    def top_predicted_words(self):
        self.args.num_examples_per_label =-1
        # if self.args.domainpmi==0:
        self.__get_traindataset()
        self.prompt_label_idx = self.train_dataset.prompt_label_idx
        self.__get_model()

        if self.args.filtering_using_prior==1 or self.args.div_prior==1:
            if self.args.domainpmi==1:
                self.__get_dominpmi_prior()
            else:
                _, all_logits_all = self.__model_forward(self.train_dataloader, dataloadertype='Unlabel_Support')
                self.__get_prior(all_logits_all)
            if self.args.filtering_using_prior==1:
                self.__dropword_with_prior()
        elif self.args.div_prior==0:
            self.prior_logits=None

        self.__get_testdataset()
        self.__get_criterion()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.test_dataloader, desc='Test')):
                logit_all = self.model(**batch)
                all_labels.append(batch['labels'])  ##0628change
                preds = self.criterion.get_top_predict_words(logit_all, self.prior_logits, self.tokenizer, batch['labels'])
                all_preds.append(preds)

        all_labels = torch.cat(all_labels, axis=0)
        all_preds = np.concatenate(all_preds)
        mic, mac = self.criterion.f1_score(all_preds, all_labels)
        log.info("mic {} mac {}".format(mic, mac))
        import pickle as pkl
        if not os.path.exists(self.args.output_dir+"/visualize_top_choice/") :
            os.mkdir(self.args.output_dir+"/visualize_top_choice/")


        with open(self.args.output_dir+"/visualize_top_choice/dataset_{}_temp_{}_seed_{}.pkl".format(self.args.dataset, self.args.template_id, self.args.seed),'wb') as fout:  # the topk predition word from the mask
            pkl.dump(self.criterion.tokens2ranks, fout)


        




        

# def print_matrix(x):
#     for i in range(x.shape[0]):
#         print([i for i in x[i]])







    




if __name__=="__main__":
    if args.task == "run_all_template":
        with open(os.path.join(args.data_dir,args.dataset, args.template_file_name),'r') as f:
            L = f.readlines()
            num_template = len([x for x in L if x!="\n"])
        mics = []
        for tempid in range(num_template):
            args.template_id = tempid
            runner = Runner(args)
            mics.append(runner.test_model_with_determined_k())
        mics = np.array(mics)
        mean, std = np.mean(mics), np.std(mics)
        log.info("mics of all template {}, mean {}, std {}".format(mics, mean, std))
    
    elif args.task == "run_all_k_value":
        with open(os.path.join(args.data_dir,args.dataset, args.template_file_name),'r') as f:
            L = f.readlines()
            num_template = len(L)
        mics = []
        for tempid in range(num_template):
            args.template_id = tempid
            runner = Runner(args)
            mics.append(runner.test_model_with_different_k())
        log.info("mics of all template and k {}".format(mics))
    
    elif args.task == "run_zero_shot_model":
        runner = Runner(args)
        mic = runner.test_model_zero_shot()
        log.info("mics {}".format(mic))
    elif args.task == "run_all_zero_shot":
        mics = {}
        with open(os.path.join(args.data_dir,args.dataset, args.template_file_name),'r') as f:
            L = f.readlines()
            num_template = len([x for x in L if x!="\n"])
        label_word_files = args.label_word_file.split(':')
        
        for tempid in range(num_template):
            mics[tempid] = {}
            args.template_id = tempid
            runner = Runner(args)
            ## test org labels
            log.info("begin test org labels")
            args.label_word_file = label_word_files[0]
            args.pred_method='none'
            args.optimize_method='none'
            mics[tempid]['none'] = []
            mics[tempid]['none'].append(runner.test_model_with_determined_k())

            log.info("begin test ranking")
            args.label_word_file = label_word_files[1]
            args.pred_method='rank'
            args.optimize_method='rank'
            mics[tempid]['rank'] = []
            mics[tempid]['rank'].append(runner.test_model_with_different_k(ks=[0.05*i for i in range(1,19)], add_auto=True))

            log.info("begin test mean")
            args.pred_method='mean'
            args.optimize_method='mean'
            mics[tempid]['mean'] = []
            mics[tempid]['mean'].append(runner.test_model_with_determined_k())
        log.info("all result of all zero-shot task:\n{}".format(mics))
    
    elif args.task == 'different_label_words_num':
        pass
    elif args.task == "top_token_visualization":
        runner = Runner(args)
        runner.top_predicted_words()
    elif args.task == "inspect_drop_words":
        runner = Runner(args)
        runner.inspect_dropped_words()

    elif args.task == "train_and_save_few_shot_model":
        runner = Runner(args)
        runner.train_model()

    elif args.task == 'run_finetuning':
        runner = Runner(args)
        runner.finetuning()

    else:
        log.error("Task not configured!")
    



    

