import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import logger
import os
log = logger.get_logger(__name__)


class BaseCriterion(nn.Module):
    def __init__(self, args):
        super(BaseCriterion, self).__init__()
        self.args = args
        pass
    
    def normalize(self, logits, priors=None):
        '''making the probability of label words across class sum to 1.
        '''
        # logits = logits/50
        # log.info("prior logit is none")
        logits_s = nn.functional.softmax(logits, dim=-1)
        # log.debug("logits_s before div {}".format(logits_s))
        if priors is not None and self.args.div_prior==1:
            priors = nn.functional.softmax(priors, dim=-1)
            # log.info("prior logit is not none")
            # log.debug("priors before div {}".format(priors))
        # else:
            # log.info("prior logit is none")
        if priors is not None and self.args.div_prior==1:
            logits_s = torch.log(logits_s/(priors+1e-15)+1e-15)
        else:
            logits_s = torch.log(logits_s+1e-15)
        return logits_s
    
    def forward(self, logit, label, prior_logits=None):
        raise NotImplementedError

    def predict(self, logits):
        raise NotImplementedError



    def evaluate(self, all_logits_all, all_prior_logits, all_labels, tokenizer):
        logits = self.get_label_words_logits(all_logits_all)
        if all_prior_logits is not None and self.args.div_prior==1:
            priors = self.get_label_words_logits(all_prior_logits)
        else:
            priors = None
        logits_s = self.normalize(logits, priors)
        all_preds =  self.predict(logits=logits_s) #np.argmax(np.max
        all_preds = all_preds.cpu().numpy()
        all_labels = all_labels.cpu().numpy()
        mi_f1 = f1_score(all_labels, all_preds, average='micro')
        ma_f1 = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)
        log.info("mi_f1 {}, ma_f1 {}, acc {}".format(mi_f1, ma_f1, acc))
        return mi_f1, ma_f1
    
    def evaluate_batch(self, all_logits_all, all_prior_logits, tokenizer):
        logits = self.get_label_words_logits(all_logits_all)
        if all_prior_logits is not None and self.args.div_prior==1:
            priors = self.get_label_words_logits(all_prior_logits)
        else:
            priors = None
        logits_s = self.normalize(logits, priors)
        all_preds =  self.predict(logits=logits_s) #np.argmax(np.max
        # self.__get_top_word(logits_s, tokenizer, all_labels, all_preds)
        all_preds = all_preds.cpu().numpy()
        return all_preds
        
        
    def f1_score(self, preds, labels):
        labels = labels.cpu().numpy()
        mi_f1 = f1_score(labels, preds, average='micro')
        ma_f1 = f1_score(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        log.info("mi_f1 {}, ma_f1 {}, acc {}".format(mi_f1, ma_f1, acc))
        return mi_f1, ma_f1
    

    





class MarginCriterion(BaseCriterion):
    def __init__(self, args, prompt_label_idx):
        super(MarginCriterion, self).__init__(args=args)
        self.prompt_label_idx = prompt_label_idx
        self.margin=self.args.margin
        self.__get_prompt_tensor(prompt_label_idx)
    
    def __get_prompt_tensor(self, prompt_label_idx):
        prompt_label_idx_one_token = [[i[0] for i in prompt_per_class] for prompt_per_class in prompt_label_idx]
        class_num = len(prompt_label_idx_one_token)
        prompt_label_tensor = torch.tensor(prompt_label_idx_one_token) #torch.zeros([class_num,max_len], dtype = torch.long)
        indexable_labels = torch.diag(torch.ones(class_num,dtype=torch.long))
        self.prompt_label_tensor = prompt_label_tensor.cuda() # (class_num, max_len)
        self.indexable_labels = indexable_labels.cuda() #(class_num, class_num*max_len)
        self.class_num = class_num

    def forward(self, logit, label, prior_logits):
        logits = self.get_label_words_logits(logit)
        if prior_logits is not None and self.args.div_prior==1:
            prior_logits = self.get_label_words_logits(prior_logits)
        logits_s = self.normalize(logits, prior_logits)
        multilabels = self.indexable_labels[label]
        mask = multilabels.unsqueeze(-1)* (1-multilabels).unsqueeze(-2)  #(batchsize, classnum*max_len, classnum*maxlen) (, i, j) is one,where i is positive id, j is negative id
        loss = torch.nn.ReLU()(mask*(self.margin+logits_s.unsqueeze(-2)-logits_s.unsqueeze(-1)))
        loss = torch.sum(loss)
        return loss

    def get_label_words_logits(self, alllogits):
        logits = alllogits[:, self.prompt_label_tensor.reshape(-1)]
        return logits

    
    def predict(self,logits):
        preds = torch.argmax(logits, dim=-1)
        return preds
    

        

class XEntCriterion(BaseCriterion):
    def __init__(self, args, prompt_label_idx):
        super(XEntCriterion, self).__init__(args=args)
        self.prompt_label_idx = prompt_label_idx
        self.margin=self.args.margin
        self.__get_prompt_tensor(prompt_label_idx)
        self.crit = nn.CrossEntropyLoss()
    
    def __get_prompt_tensor(self, prompt_label_idx):
        prompt_label_idx_one_token = [[i[0] for i in prompt_per_class] for prompt_per_class in prompt_label_idx]
        class_num = len(prompt_label_idx_one_token)
        prompt_label_tensor = torch.tensor(prompt_label_idx_one_token) #torch.zeros([class_num,max_len], dtype = torch.long)
        indexable_labels = torch.diag(torch.ones(class_num,dtype=torch.long))
        self.prompt_label_tensor = prompt_label_tensor.cuda() # (class_num, max_len)
        self.indexable_labels = indexable_labels.cuda() #(class_num, class_num*max_len)
        self.class_num = class_num
    
    def forward(self, logit, label, prior_logits):
        logits = self.get_label_words_logits(logit)
        if prior_logits is not None and self.args.div_prior==1:
            prior_logits = self.get_label_words_logits(prior_logits)
        logits_s = self.normalize(logits, prior_logits)
        loss = self.crit(logits_s, label)
        return loss
    
    def predict(self,logits):
        preds = torch.argmax(logits, dim=-1)
        return preds
    
    def get_label_words_logits(self, alllogits):
        logits = alllogits[:, self.prompt_label_tensor.reshape(-1)]
        return logits
    

        

# class LinearXEntCriterion(BaseCriterion):
#     def __init__(self, args, prompt_label_idx):
#         super(LinearXEntCriterion, self).__init__(args=args)
#         self.bce_criterion = torch.nn.BCEWithLogitsLoss()
#         self.__get_indexable_labels(prompt_label_idx)

#     def forward(self, logit, label, prior_logits):
#         logits = []
#         priors = []
#         for prompt_label_per_class in self.prompt_label_idx:
#             for words in prompt_label_per_class:
#                 logits.append(logit[:, words].mean(dim=-1))
#                 priors.append(prior_logits[:, words].mean(dim=-1))
#         logits = torch.vstack(logits).T
#         priors = torch.vstack(priors).T
#         multilabels = self.indexable_labels[label]
#         logits_s=logits-priors
        
#         loss = self.bce_criterion(logits_s, multilabels.to(torch.float))*0.01

#         return loss

# class MeanCriterion(BaseCriterion):
#     def __init__(self, args, prompt_label_idx):
#         super(MeanCriterion, self).__init__(args=args)
#         self.bce_criterion = torch.nn.BCEWithLogitsLoss()
#         self.__get_indexable_labels(prompt_label_idx)
    
#     def forward(self, logit, label, prior_logits):
#         logit = [ torch.mean(logit, dim=-1)]
#         loss = self.crit(logit, label)
#         return loss, 

class RankingCriterion(BaseCriterion):
    def __init__(self, args, prompt_label_idx):
        super(RankingCriterion, self).__init__(args = args)
        self.prompt_label_idx = prompt_label_idx
        self.margin=self.args.margin
        self.register_buffer('topk',torch.tensor(0).cuda())
        if self.args.multitoken==1:
            log.info("register multitoken function")
            self.get_label_words_logits = self.get_label_words_logits_multitoken
            self.__get_prompt_tensor_and_mask = self.__get_prompt_tensor_and_mask_multitoken
        else:
            log.info("register singletoken function")
            self.get_label_words_logits = self.get_label_words_logits_singletoken
            self.__get_prompt_tensor_and_mask = self.__get_prompt_tensor_and_mask_singletoken
        self.__get_prompt_tensor_and_mask(prompt_label_idx)

        self.dropout = nn.Dropout(0.2)
            
    def __get_prompt_tensor_and_mask_singletoken(self, prompt_label_idx):
        prompt_label_idx_one_token = [[i[0] for i in prompt_per_class] for prompt_per_class in prompt_label_idx]
        class_num = len(prompt_label_idx_one_token)
        max_len  = max([len(x) for x in prompt_label_idx_one_token])
        log.info("xxxx {} {}".format([len(x) for x in prompt_label_idx],[len(x) for x in prompt_label_idx_one_token] ))
        prompt_label_tensor = torch.zeros([class_num,max_len], dtype = torch.long)
        prompt_label_mask =  torch.zeros_like(prompt_label_tensor)
        indexable_labels = torch.zeros([class_num,max_len*class_num],dtype=torch.long)
        for id, prompt_per_class in enumerate(prompt_label_idx_one_token):
            prompt_label_tensor[id, :len(prompt_per_class)] = torch.tensor(prompt_per_class).to(torch.long)
            prompt_label_mask[id, :len(prompt_per_class)] = 1
            indexable_labels[id, id*max_len:(id*max_len+len(prompt_per_class))] = 1
        self.prompt_label_tensor = prompt_label_tensor.cuda() # (class_num, max_len)
        self.prompt_label_mask = prompt_label_mask.cuda() #(class_num, max_len)
        self.indexable_labels = indexable_labels.cuda() #(class_num, class_num*max_len)
        self.prompt_maskid = torch.where(self.prompt_label_mask.reshape(-1)==0)[0] #  1-dim, if prompt_label_tensor[rowid, colid]==0, then row_id*max_len+col_id in this tensor 
        self.label_space_size = sum([len(x) for x in prompt_label_idx_one_token]) # total number of label words
        self.class_num = class_num
        self.max_len =  max_len

        self.learnable_label_weights = torch.zeros([class_num, max_len]).cuda()
        self.learnable_label_weights = self.learnable_label_weights.reshape(-1)
        self.learnable_label_weights[self.prompt_maskid]-=1000
        self.learnable_label_weights = self.learnable_label_weights.reshape([class_num, max_len])
        self.learnable_label_weights = torch.nn.parameter.Parameter(self.learnable_label_weights, requires_grad=True)

    def __get_prompt_tensor_and_mask_multitoken(self, prompt_label_idx):
        prompt_label_idx_multitoken = [[i for i in prompt_per_class] for prompt_per_class in prompt_label_idx]
        class_num = len(prompt_label_idx_multitoken)
        max_len  = max([len(x) for x in prompt_label_idx_multitoken])
        max_token_num = max([len(y)  for x in prompt_label_idx_multitoken for y in x])
        prompt_label_tensor = torch.zeros([class_num,max_len, max_token_num], dtype = torch.long)
        prompt_label_mask_3d =  torch.zeros_like(prompt_label_tensor)
        indexable_labels = torch.zeros([class_num,max_len*class_num],dtype=torch.long)
        for id, prompt_per_class in enumerate(prompt_label_idx_multitoken):
            for j, label_word in enumerate(prompt_per_class):
                prompt_label_tensor[id, j, :len(label_word)] = torch.tensor(label_word).to(torch.long)
                prompt_label_mask_3d[id, j, :len(label_word)] = 1
            indexable_labels[id, id*max_len:(id*max_len+len(prompt_per_class))] = 1
        self.prompt_label_tensor = prompt_label_tensor.cuda() # (class_num, max_len)
        self.prompt_label_mask_3d = prompt_label_mask_3d.cuda() #(class_num, max_len)
        self.prompt_label_mask = (prompt_label_mask_3d.sum(dim=-1)>0).to(torch.long).cuda()
        self.indexable_labels = indexable_labels.cuda() #(class_num, class_num*max_len)
        self.prompt_maskid = torch.where(self.prompt_label_mask.reshape(-1)==0)[0] #  1-dim, if prompt_label_tensor[rowid, colid]==0, then row_id*max_len+col_id in this tensor 
        self.label_space_size = sum([len(x) for x in prompt_label_idx_multitoken]) # total number of label words
        self.class_num = class_num
        self.max_len =  max_len
        self.max_token_num = max_token_num
        # log.info("max_token_num {} prompt_maskid {} {}".format(max_token_num, self.prompt_maskid, self.prompt_label_mask))

        self.learnable_label_weights = torch.zeros([class_num, max_len]).cuda()
        self.learnable_label_weights = self.learnable_label_weights.reshape(-1)
        self.learnable_label_weights[self.prompt_maskid]-=1000
        self.learnable_label_weights = self.learnable_label_weights.reshape([class_num, max_len])
        self.learnable_label_weights = torch.nn.parameter.Parameter(self.learnable_label_weights, requires_grad=True)


    def get_label_words_logits_singletoken(self, alllogits):
        logits = alllogits[:, self.prompt_label_tensor.reshape(-1)]

        logits[:, self.prompt_maskid]-=1000
        # log.debug("ssfasf {}".format(logits[:, 742]))
        return logits
    
    # def train(self):
    #     self.dropout = nn.Dropout(0)
    
    # def eval(self):
    #     self.dropout = nn.Dropout(0)

    def get_label_words_logits_multitoken(self, alllogits):
        logits = alllogits[:, self.prompt_label_tensor.reshape(-1)].reshape(alllogits.size(0), self.class_num, self.max_len,self.max_token_num)
        logits = (logits*self.prompt_label_mask_3d.unsqueeze(0)).sum(dim=-1)
        # log.info("logits {} {} {}".format(logits, logits.size(), (self.prompt_label_mask_3d.sum(dim=-1).unsqueeze(0))))
        logits = logits/(self.prompt_label_mask_3d.sum(dim=-1).unsqueeze(0)+1e-15)
        logits = logits.reshape(alllogits.size(0),-1)
        logits[:, self.prompt_maskid]-=1000
        # log.info("logits aa {}".format(logits))
        return logits


    def forward(self, logit, label, prior_logits=None):
        logits = self.get_label_words_logits(logit)
        if prior_logits is not None and self.args.div_prior==1:
            prior_logits = self.get_label_words_logits(prior_logits)
        logits_s = self.normalize(logits, prior_logits)
        # log.debug("logits_s {}".format(logits_s))
        multilabels = self.indexable_labels[label]
        mm = self.prompt_label_mask.reshape(-1).repeat(len(multilabels),1)
        mask = multilabels.unsqueeze(-1)* (mm-multilabels).unsqueeze(-2)  #(batchsize, classnum*max_len, classnum*maxlen) (, i, j) is one,where i is positive id, j is negative id
        weight = nn.functional.softmax(self.learnable_label_weights,dim=-1).reshape(-1).unsqueeze(0).unsqueeze(-1)  # (1, classnum*maxlen, 1] 
        mid = mask*(self.margin+logits_s.unsqueeze(-2)-logits_s.unsqueeze(-1))
        # mid = self.dropout(mid)
        loss_ = torch.nn.ReLU()(mid)
        loss = weight*loss_
        loss = torch.sum(loss)
        return loss


    def __get_weights(self):
        weight = nn.functional.softmax(self.args.pred_temperature*self.learnable_label_weights,dim=-1).reshape(-1)
        self.weights = weight#.cpu().tolist()
        return self.weights

    def __get_top_word(self, logits, tokenizer, labels, preds):
        # log.debug("ssfasf {}".format(logits[:, 742]))
        
        def logitid2promptid(x):
            rowid = x//self.max_len
            colid = x-x//self.max_len*self.max_len
            return rowid, colid
        
        idxs = torch.argsort(-logits, dim=-1)[:,:10].detach().cpu().tolist()
        tokens = []
        for rowid, rows in enumerate(idxs):
            tokens.append([])
            for x in rows:
                
                rowid, colid = logitid2promptid(x)
                weight = self.weights[x].item()
                # log.info("row id cow id {} {} {} {}".format(x, rowid, colid, logits[rowid][x]))
                try:
                    token = [weight, tokenizer.convert_ids_to_tokens(self.prompt_label_idx[rowid][colid])]
                except:
                    log.info("row id cow id {} {} {} {} | {} {}".format(x, rowid, colid, torch.sort(-logits[rowid]).values.cpu().tolist(), self.prompt_label_mask[rowid, colid], x in self.prompt_maskid.detach().cpu().numpy()))
                    
                    exit()

                tokens[-1].append(token)
        for id, token in enumerate(tokens):
            log.debug("GT:{} PR:{} | {}".format(labels[id],  preds[id], token))

    # def remove_label_word_based_on_validation_set(self, logits, tokenizer, labels, preds):
        # idxs = torch.argsort(-logits, dim=-1)[:,:10].detach().cpu().tolist()
        # idxs 


    def predict(self,logits):
        self.__get_weights()
        # log.debug("{}".format(self.weights))
        # keepids = torch.where(self.weights> 1e-5):
        # self.weights =self.weights[keepids]
        # logits = logits[:, keepids]
        # logit

        nottopk = torch.argsort(-logits, dim=-1)[:,self.topk:]
        rowid = torch.arange(nottopk.size(0)).unsqueeze(-1).expand_as(nottopk)
        index = (rowid.reshape(-1), nottopk.reshape(-1))
        scores = torch.clone(self.weights).unsqueeze(0).repeat(logits.size(0),1)
        scores[index]=0
        scores = scores.reshape([logits.size(0), self.class_num, self.max_len])
        scores = torch.sum(scores, dim=-1)
        preds = torch.argmax(scores, dim=-1)
        # print(preds)
        return preds


    def set_k_for_ranking(self,topkratio=0, verbose=True):
        assert self.args.pred_method=='rank'
        if topkratio > 0:
            self.topk = torch.tensor(int(self.label_space_size*topkratio))
        else:
            log.error("Not configure k value for ranking")
        if verbose:
            log.info("Using k={}, k-ratio={}".format(self.topk, topkratio))


    def evaluate(self, all_logits_all, all_prior_logits, all_labels, tokenizer):
        logits = self.get_label_words_logits(all_logits_all)
        if all_prior_logits is not None and self.args.div_prior==1:
            priors = self.get_label_words_logits(all_prior_logits)
        else:
            priors = None
        logits_s = self.normalize(logits, priors)
        all_preds =  self.predict(logits=logits_s) #np.argmax(np.max
        # self.__get_top_word(logits_s, tokenizer, all_labels, all_preds)
        all_preds = all_preds.cpu().numpy()
        all_labels = all_labels.cpu().numpy()
        mi_f1 = f1_score(all_labels, all_preds, average='micro')
        ma_f1 = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)
        log.info("mi_f1 {}, ma_f1 {}, acc {}".format(mi_f1, ma_f1, acc))
        return mi_f1, ma_f1
    
    def evaluate_batch(self, all_logits_all, all_prior_logits, tokenizer):
        logits = self.get_label_words_logits(all_logits_all)
        if all_prior_logits is not None and self.args.div_prior==1:
            priors = self.get_label_words_logits(all_prior_logits)
        else:
            priors = None
        logits_s = self.normalize(logits, priors)
        all_preds =  self.predict(logits=logits_s) #np.argmax(np.max
        # self.__get_top_word(logits_s, tokenizer, all_labels, all_preds)
        all_preds = all_preds.cpu().numpy()
        return all_preds
        
        
    def f1_score(self, preds, labels):
        labels = labels.cpu().numpy()
        mi_f1 = f1_score(labels, preds, average='micro')
        ma_f1 = f1_score(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        log.info("mi_f1 {}, ma_f1 {}, acc {}".format(mi_f1, ma_f1, acc))
        return mi_f1, ma_f1
    
    def determine_k(self, all_logits_all, all_prior_logits=None, labels=None):
        logits = self.get_label_words_logits(all_logits_all)
        if all_prior_logits is not None and self.args.div_prior==1:
            priors = self.get_label_words_logits(all_prior_logits)
        else:
            priors = None
        logits_s = self.normalize(logits, priors) ##

        if labels is not None:
            labels = labels.cpu().numpy()
        ks = [0.02*i for i in range(1,50)]
        preds = []
        mics = []
        for k in ks:
            self.set_k_for_ranking(topkratio=k, verbose=False)
            pred = self.predict(logits=logits_s)
            if labels is not None:
                pred = pred.cpu().numpy()
                mi_f1 = f1_score(labels, pred, average='micro')
                mics.append(mi_f1)
            preds.append(pred)
        if len(mics)>0:
            mics = np.array(mics)
            maxmic = np.max(mics)
            best_ids = np.where(mics==maxmic)[0]
            best_id = best_ids[0] ## get the first middle point
            # best_mic =  mics[best_id]
            best_k = ks[best_id]
            return best_k
    
    def determine_k_base_on_unlabeled_corpus(self, all_logits_all, all_prior_logits=None):
        logits = self.get_label_words_logits(all_logits_all)
        if all_prior_logits is not None and self.args.div_prior==1:
            priors = self.get_label_words_logits(all_prior_logits)
        else:
            priors = None
        logits_s = self.normalize(logits, priors) ##

        # last_preds = np.zeros(logits.shape[0])
        # flip_rec = []
        all_preds = []
        ks = np.linspace(0,1,101)
        for id, k in enumerate(ks):
            self.set_k_for_ranking(topkratio=k, verbose=False)
            preds = self.predict(logits=logits_s)
            all_preds.append(preds.detach().cpu().numpy())
        all_preds = np.vstack(all_preds)
        best_k_pos = self.__get_stable_interval(all_preds, orders=[1,2,3,4])
        best_k = ks[best_k_pos]
        log.info("best k {}".format(best_k))
        return best_k

    
    
    def __get_stable_interval(self, x, orders):
        # print(x, orders)
        def num_flip(a,b):
            return np.sum((a!=b).astype(np.int))

        # hi = orders[-1]
        stable_nums = []
        nf_orders = {}
        for j in orders:
            nf_orders[j] = []
            for i in range(x.shape[0]):
                if j<=i and i+j<x.shape[0]:
                    n1 = num_flip(x[i-j], x[i])
                    n2 = num_flip(x[i+j], x[i])
                    n = (n1+n2)/2
                    nf_orders[j].append(n)
        # print(nf_orders)
        ret = [nf_orders[od][orders[-1]-od:od-orders[-1]] for od in orders[:-1]]
        ret.append(nf_orders[orders[-1]])

        ret = np.array(ret)
        # print(ret)
        ret = ret.sum(axis=0)
        # print(ret)

        best_pos = np.argmin(ret)+orders[-1]
        # print(best_pos)
        # exit()
        return best_pos
            
        


        
class LinearXEntCriterion(BaseCriterion):
    def __init__(self, args, prompt_label_idx):
        super(LinearXEntCriterion, self).__init__(args = args)
        self.prompt_label_idx = prompt_label_idx
        self.margin=self.args.margin
        self.register_buffer('topk',torch.tensor(0).cuda())
        if self.args.multitoken==1:
            log.info("register multitoken function")
            self.get_label_words_logits = self.get_label_words_logits_multitoken
            self.__get_prompt_tensor_and_mask = self.__get_prompt_tensor_and_mask_multitoken
        else:
            log.info("register singletoken function")
            self.get_label_words_logits = self.get_label_words_logits_singletoken
            self.__get_prompt_tensor_and_mask = self.__get_prompt_tensor_and_mask_singletoken
        self.__get_prompt_tensor_and_mask(prompt_label_idx)

        self.dropout = nn.Dropout(0.2)
        self.crit = nn.CrossEntropyLoss()
            
    def __get_prompt_tensor_and_mask_singletoken(self, prompt_label_idx):
        prompt_label_idx_one_token = [[i[0] for i in prompt_per_class] for prompt_per_class in prompt_label_idx]
        class_num = len(prompt_label_idx_one_token)
        max_len  = max([len(x) for x in prompt_label_idx_one_token])
        log.info("xxxx {} {}".format([len(x) for x in prompt_label_idx],[len(x) for x in prompt_label_idx_one_token] ))
        prompt_label_tensor = torch.zeros([class_num,max_len], dtype = torch.long)
        prompt_label_mask =  torch.zeros_like(prompt_label_tensor)
        indexable_labels = torch.zeros([class_num,max_len*class_num],dtype=torch.long)
        for id, prompt_per_class in enumerate(prompt_label_idx_one_token):
            prompt_label_tensor[id, :len(prompt_per_class)] = torch.tensor(prompt_per_class).to(torch.long)
            prompt_label_mask[id, :len(prompt_per_class)] = 1
            indexable_labels[id, id*max_len:(id*max_len+len(prompt_per_class))] = 1
        self.prompt_label_tensor = prompt_label_tensor.cuda() # (class_num, max_len)
        self.prompt_label_mask = prompt_label_mask.cuda() #(class_num, max_len)
        self.indexable_labels = indexable_labels.cuda() #(class_num, class_num*max_len)
        self.prompt_maskid = torch.where(self.prompt_label_mask.reshape(-1)==0)[0] #  1-dim, if prompt_label_tensor[rowid, colid]==0, then row_id*max_len+col_id in this tensor 
        self.label_space_size = sum([len(x) for x in prompt_label_idx_one_token]) # total number of label words
        self.class_num = class_num
        self.max_len =  max_len

        self.learnable_label_weights = torch.zeros([class_num, max_len]).cuda()
        self.learnable_label_weights = self.learnable_label_weights.reshape(-1)
        self.learnable_label_weights[self.prompt_maskid]-=1000
        self.learnable_label_weights = self.learnable_label_weights.reshape([class_num, max_len])
        self.learnable_label_weights = torch.nn.parameter.Parameter(self.learnable_label_weights, requires_grad=True)

    def __get_prompt_tensor_and_mask_multitoken(self, prompt_label_idx):
        prompt_label_idx_multitoken = [[i for i in prompt_per_class] for prompt_per_class in prompt_label_idx]
        class_num = len(prompt_label_idx_multitoken)
        max_len  = max([len(x) for x in prompt_label_idx_multitoken])
        max_token_num = max([len(y)  for x in prompt_label_idx_multitoken for y in x])
        prompt_label_tensor = torch.zeros([class_num,max_len, max_token_num], dtype = torch.long)
        prompt_label_mask_3d =  torch.zeros_like(prompt_label_tensor)
        indexable_labels = torch.zeros([class_num,max_len*class_num],dtype=torch.long)
        for id, prompt_per_class in enumerate(prompt_label_idx_multitoken):
            for j, label_word in enumerate(prompt_per_class):
                prompt_label_tensor[id, j, :len(label_word)] = torch.tensor(label_word).to(torch.long)
                prompt_label_mask_3d[id, j, :len(label_word)] = 1
            indexable_labels[id, id*max_len:(id*max_len+len(prompt_per_class))] = 1
        self.prompt_label_tensor = prompt_label_tensor.cuda() # (class_num, max_len)
        self.prompt_label_mask_3d = prompt_label_mask_3d.cuda() #(class_num, max_len)
        self.prompt_label_mask = (prompt_label_mask_3d.sum(dim=-1)>0).to(torch.long).cuda()
        self.indexable_labels = indexable_labels.cuda() #(class_num, class_num*max_len)
        self.prompt_maskid = torch.where(self.prompt_label_mask.reshape(-1)==0)[0] #  1-dim, if prompt_label_tensor[rowid, colid]==0, then row_id*max_len+col_id in this tensor 
        self.label_space_size = sum([len(x) for x in prompt_label_idx_multitoken]) # total number of label words
        self.class_num = class_num
        self.max_len =  max_len
        self.max_token_num = max_token_num
        # log.info("max_token_num {} prompt_maskid {} {}".format(max_token_num, self.prompt_maskid, self.prompt_label_mask))

        self.learnable_label_weights = torch.zeros([class_num, max_len]).cuda()
        self.learnable_label_weights = self.learnable_label_weights.reshape(-1)
        self.learnable_label_weights[self.prompt_maskid]-=1000
        self.learnable_label_weights = self.learnable_label_weights.reshape([class_num, max_len])
        self.learnable_label_weights = torch.nn.parameter.Parameter(self.learnable_label_weights, requires_grad=True)


    def get_label_words_logits_singletoken(self, alllogits):
        logits = alllogits[:, self.prompt_label_tensor.reshape(-1)]

        logits[:, self.prompt_maskid]-=1000
        # log.debug("ssfasf {}".format(logits[:, 742]))
        return logits


    def get_label_words_logits_multitoken(self, alllogits):
        logits = alllogits[:, self.prompt_label_tensor.reshape(-1)].reshape(alllogits.size(0), self.class_num, self.max_len,self.max_token_num)
        logits = (logits*self.prompt_label_mask_3d.unsqueeze(0)).sum(dim=-1)
        # log.info("logits {} {} {}".format(logits, logits.size(), (self.prompt_label_mask_3d.sum(dim=-1).unsqueeze(0))))
        logits = logits/(self.prompt_label_mask_3d.sum(dim=-1).unsqueeze(0)+1e-15)
        logits = logits.reshape(alllogits.size(0),-1)
        logits[:, self.prompt_maskid]-=1000
        # log.info("logits aa {}".format(logits))
        return logits


    def forward(self, logit, label, prior_logits=None):
        logits = self.get_label_words_logits(logit)
        if prior_logits is not None and self.args.div_prior==1:
            prior_logits = self.get_label_words_logits(prior_logits)
        logits_s = self.normalize(logits, prior_logits)
        # log.debug("logits_s {}".format(logits_s))
        multilabels = self.indexable_labels[label]

        weights = nn.functional.softmax(self.learnable_label_weights,dim=-1).reshape(-1).unsqueeze(0)
        mm = self.prompt_label_mask.reshape(-1).repeat(len(multilabels),1)
        # log.debug("qqqqqq{} {}".format(mm.size(), weights.size()))

        logits_soft = (logits_s*mm*weights).reshape(logit.size(0),self.class_num,self.max_len)
        logits_soft= torch.sum(logits_soft, dim=-1)
        loss = self.crit(logits_soft, label)
        # exit()
        return loss


 


    def predict(self,logits):
        weights = nn.functional.softmax(self.learnable_label_weights,dim=-1).reshape(-1).unsqueeze(0)
        mm = self.prompt_label_mask.reshape(-1).repeat(logits.size(0),1)
        logits_soft = (logits*mm*weights).reshape(logits.size(0),self.class_num,self.max_len)
        logits_soft= torch.sum(logits_soft, dim=-1)
        preds = torch.argmax(logits_soft, dim=-1)
        # print(preds)
        return preds


 

    def evaluate(self, all_logits_all, all_prior_logits, all_labels, tokenizer):
        logits = self.get_label_words_logits(all_logits_all)
        if all_prior_logits is not None and self.args.div_prior==1:
            priors = self.get_label_words_logits(all_prior_logits)
        else:
            priors = None
        logits_s = self.normalize(logits, priors)
        all_preds =  self.predict(logits=logits_s) #np.argmax(np.max
        # self.__get_top_word(logits_s, tokenizer, all_labels, all_preds)
        all_preds = all_preds.cpu().numpy()
        all_labels = all_labels.cpu().numpy()
        mi_f1 = f1_score(all_labels, all_preds, average='micro')
        ma_f1 = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)
        log.info("mi_f1 {}, ma_f1 {}, acc {}".format(mi_f1, ma_f1, acc))
        return mi_f1, ma_f1
    
    def evaluate_batch(self, all_logits_all, all_prior_logits, tokenizer):
        logits = self.get_label_words_logits(all_logits_all)
        if all_prior_logits is not None and self.args.div_prior==1:
            priors = self.get_label_words_logits(all_prior_logits)
        else:
            priors = None
        logits_s = self.normalize(logits, priors)
        all_preds =  self.predict(logits=logits_s) #np.argmax(np.max
        # self.__get_top_word(logits_s, tokenizer, all_labels, all_preds)
        all_preds = all_preds.cpu().numpy()
        return all_preds
        
        
    def f1_score(self, preds, labels):
        labels = labels.cpu().numpy()
        mi_f1 = f1_score(labels, preds, average='micro')
        ma_f1 = f1_score(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        log.info("mi_f1 {}, ma_f1 {}, acc {}".format(mi_f1, ma_f1, acc))
        return mi_f1, ma_f1
    




        


       
class MeanCriterion(BaseCriterion):
    def __init__(self, args, prompt_label_idx):
        super(MeanCriterion, self).__init__(args = args)
        self.prompt_label_idx = prompt_label_idx
        self.margin=self.args.margin
        if self.args.multitoken==1:
            log.info("register multitoken function")
            self.get_label_words_logits = self.get_label_words_logits_multitoken
            self.__get_prompt_tensor_and_mask = self.__get_prompt_tensor_and_mask_multitoken
        else:
            log.info("register singletoken function")
            self.get_label_words_logits = self.get_label_words_logits_singletoken
            self.__get_prompt_tensor_and_mask = self.__get_prompt_tensor_and_mask_singletoken
        self.__get_prompt_tensor_and_mask(prompt_label_idx)

        self.crit = nn.CrossEntropyLoss()
            
    def __get_prompt_tensor_and_mask_singletoken(self, prompt_label_idx):
        prompt_label_idx_one_token = [[i[0] for i in prompt_per_class] for prompt_per_class in prompt_label_idx]
        class_num = len(prompt_label_idx_one_token)
        max_len  = max([len(x) for x in prompt_label_idx_one_token])
        log.info("xxxx {} {}".format([len(x) for x in prompt_label_idx],[len(x) for x in prompt_label_idx_one_token] ))
        prompt_label_tensor = torch.zeros([class_num,max_len], dtype = torch.long)
        prompt_label_mask =  torch.zeros_like(prompt_label_tensor)
        indexable_labels = torch.zeros([class_num,max_len*class_num],dtype=torch.long)
        for id, prompt_per_class in enumerate(prompt_label_idx_one_token):
            prompt_label_tensor[id, :len(prompt_per_class)] = torch.tensor(prompt_per_class).to(torch.long)
            prompt_label_mask[id, :len(prompt_per_class)] = 1
            indexable_labels[id, id*max_len:(id*max_len+len(prompt_per_class))] = 1
        self.prompt_label_tensor = prompt_label_tensor.cuda() # (class_num, max_len)
        self.prompt_label_mask = prompt_label_mask.cuda() #(class_num, max_len)
        self.indexable_labels = indexable_labels.cuda() #(class_num, class_num*max_len)
        self.prompt_maskid = torch.where(self.prompt_label_mask.reshape(-1)==0)[0] #  1-dim, if prompt_label_tensor[rowid, colid]==0, then row_id*max_len+col_id in this tensor 
        self.label_space_size = sum([len(x) for x in prompt_label_idx_one_token]) # total number of label words
        self.class_num = class_num
        self.max_len =  max_len

        self.learnable_label_weights = torch.zeros([class_num, max_len]).cuda()
        self.learnable_label_weights = self.learnable_label_weights.reshape(-1)
        self.learnable_label_weights[self.prompt_maskid]-=1000
        self.learnable_label_weights = self.learnable_label_weights.reshape([class_num, max_len])
        self.learnable_label_weights = torch.nn.parameter.Parameter(self.learnable_label_weights, requires_grad=True)

    def __get_prompt_tensor_and_mask_multitoken(self, prompt_label_idx):
        prompt_label_idx_multitoken = [[i for i in prompt_per_class] for prompt_per_class in prompt_label_idx]
        class_num = len(prompt_label_idx_multitoken)
        max_len  = max([len(x) for x in prompt_label_idx_multitoken])
        max_token_num = max([len(y)  for x in prompt_label_idx_multitoken for y in x])
        prompt_label_tensor = torch.zeros([class_num,max_len, max_token_num], dtype = torch.long)
        prompt_label_mask_3d =  torch.zeros_like(prompt_label_tensor)
        indexable_labels = torch.zeros([class_num,max_len*class_num],dtype=torch.long)
        for id, prompt_per_class in enumerate(prompt_label_idx_multitoken):
            for j, label_word in enumerate(prompt_per_class):
                prompt_label_tensor[id, j, :len(label_word)] = torch.tensor(label_word).to(torch.long)
                prompt_label_mask_3d[id, j, :len(label_word)] = 1
            indexable_labels[id, id*max_len:(id*max_len+len(prompt_per_class))] = 1
        self.prompt_label_tensor = prompt_label_tensor.cuda() # (class_num, max_len)
        self.prompt_label_mask_3d = prompt_label_mask_3d.cuda() #(class_num, max_len)
        self.prompt_label_mask = (prompt_label_mask_3d.sum(dim=-1)>0).to(torch.long).cuda()
        self.indexable_labels = indexable_labels.cuda() #(class_num, class_num*max_len)
        self.prompt_maskid = torch.where(self.prompt_label_mask.reshape(-1)==0)[0] #  1-dim, if prompt_label_tensor[rowid, colid]==0, then row_id*max_len+col_id in this tensor 
        self.label_space_size = sum([len(x) for x in prompt_label_idx_multitoken]) # total number of label words
        self.class_num = class_num
        self.max_len =  max_len
        self.max_token_num = max_token_num
        # log.info("max_token_num {} prompt_maskid {} {}".format(max_token_num, self.prompt_maskid, self.prompt_label_mask))

        self.learnable_label_weights = torch.zeros([class_num, max_len]).cuda()
        self.learnable_label_weights = self.learnable_label_weights.reshape(-1)
        self.learnable_label_weights[self.prompt_maskid]-=1000
        self.learnable_label_weights = self.learnable_label_weights.reshape([class_num, max_len])
        self.learnable_label_weights = torch.nn.parameter.Parameter(self.learnable_label_weights, requires_grad=True)


    def get_label_words_logits_singletoken(self, alllogits):
        logits = alllogits[:, self.prompt_label_tensor.reshape(-1)]

        logits[:, self.prompt_maskid]-=1000
        # log.debug("ssfasf {}".format(logits[:, 742]))
        return logits


    def get_label_words_logits_multitoken(self, alllogits):
        logits = alllogits[:, self.prompt_label_tensor.reshape(-1)].reshape(alllogits.size(0), self.class_num, self.max_len,self.max_token_num)
        logits = (logits*self.prompt_label_mask_3d.unsqueeze(0)).sum(dim=-1)
        # log.info("logits {} {} {}".format(logits, logits.size(), (self.prompt_label_mask_3d.sum(dim=-1).unsqueeze(0))))
        logits = logits/(self.prompt_label_mask_3d.sum(dim=-1).unsqueeze(0)+1e-15)
        logits = logits.reshape(alllogits.size(0),-1)
        logits[:, self.prompt_maskid]-=1000
        # log.info("logits aa {}".format(logits))
        return logits


    # def forward(self, logit, label, prior_logits=None):
    #     logits = self.get_label_words_logits(logit)
    #     if prior_logits is not None and self.args.div_prior==1:
    #         prior_logits = self.get_label_words_logits(prior_logits)
    #     logits_s = self.normalize(logits, prior_logits)
    #     # log.debug("logits_s {}".format(logits_s))
    #     multilabels = self.indexable_labels[label]

    #     weights = nn.functional.softmax(self.learnable_label_weights,dim=-1).reshape(-1).unsqueeze(0)
    #     mm = self.prompt_label_mask.reshape(-1).repeat(len(multilabels),1)
    #     # log.debug("qqqqqq{} {}".format(mm.size(), weights.size()))

    #     logits_soft = (logits_s*mm*weights).reshape(logit.size(0),self.class_num,self.max_len)
    #     logits_soft= torch.sum(logits_soft, dim=-1)
    #     loss = self.crit(logits_soft, label)
    #     # exit()
    #     return loss


 


    def predict(self,logits):
        # weights = nn.functional.softmax(self.learnable_label_weights,dim=-1).reshape(-1).unsqueeze(0)
        mm = self.prompt_label_mask.reshape(-1).repeat(logits.size(0),1)
        logits_soft = (logits*mm).reshape(logits.size(0),self.class_num,self.max_len)
        logits_soft= torch.sum(logits_soft, dim=-1)/torch.sum(self.prompt_label_mask,dim=-1)
        preds = torch.argmax(logits_soft, dim=-1)
        # print(preds)
        return preds


 

    def evaluate(self, all_logits_all, all_prior_logits, all_labels, tokenizer):
        logits = self.get_label_words_logits(all_logits_all)
        if all_prior_logits is not None and self.args.div_prior==1:
            priors = self.get_label_words_logits(all_prior_logits)
        else:
            priors = None
        logits_s = self.normalize(logits, priors)
        all_preds =  self.predict(logits=logits_s) #np.argmax(np.max
        # self.__get_top_word(logits_s, tokenizer, all_labels, all_preds)
        all_preds = all_preds.cpu().numpy()
        all_labels = all_labels.cpu().numpy()
        mi_f1 = f1_score(all_labels, all_preds, average='micro')
        ma_f1 = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)
        log.info("mi_f1 {}, ma_f1 {}, acc {}".format(mi_f1, ma_f1, acc))
        return mi_f1, ma_f1
    
    def evaluate_batch(self, all_logits_all, all_prior_logits, tokenizer):
        logits = self.get_label_words_logits(all_logits_all)
        if all_prior_logits is not None and self.args.div_prior==1:
            priors = self.get_label_words_logits(all_prior_logits)
        else:
            priors = None
        logits_s = self.normalize(logits, priors)
        all_preds =  self.predict(logits=logits_s) #np.argmax(np.max
        # self.__get_top_word(logits_s, tokenizer, all_labels, all_preds)
        all_preds = all_preds.cpu().numpy()
        return all_preds
        
        
    def f1_score(self, preds, labels):
        labels = labels.cpu().numpy()
        mi_f1 = f1_score(labels, preds, average='micro')
        ma_f1 = f1_score(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        log.info("mi_f1 {}, ma_f1 {}, acc {}".format(mi_f1, ma_f1, acc))
        return mi_f1, ma_f1

    def get_top_predict_words(self, all_logits_all, all_prior_logits, tokenizer, labels):
        logits = self.get_label_words_logits(all_logits_all)
        if all_prior_logits is not None and self.args.div_prior==1:
            priors = self.get_label_words_logits(all_prior_logits)
        else:
            priors = None
        logits_s = self.normalize(logits, priors)
        all_preds =  self.predict(logits=logits_s).cpu().tolist()
        labels = labels.cpu().tolist()
        def logitid2promptid(x):
            rowid = x//self.max_len
            colid = x-x//self.max_len*self.max_len
            return rowid, colid
        
        idxs = torch.argsort(-logits_s, dim=-1).detach().cpu().tolist()
        # tokens = []

        if not hasattr(self, 'tokens2ranks'):
            self.tokens2ranks = {}
            self.glbrowid = 0
            log.debug("build attributes tokens2ranks")
        # else:
            # log.debug("---{} {}".format(list(self.tokens2ranks.keys())[0], self.tokens2ranks[list(self.tokens2ranks.keys())[0]]))

        for rowid, rows in enumerate(idxs):
            for rank, x in enumerate(rows):    
                classrowid, colid = logitid2promptid(x)
                # weight = self.weights[x].item()
                try:
                    prompt_idx = self.prompt_label_idx[classrowid][colid]
                except IndexError:
                    continue
                    
                token = " ".join(tokenizer.convert_ids_to_tokens(prompt_idx))

                savecontent = [rank, all_preds[rowid], labels[rowid], self.glbrowid]
                if token in self.tokens2ranks:
                    self.tokens2ranks[token].append(savecontent)
                else:
                    self.tokens2ranks[token] = [savecontent]
            self.glbrowid+=1
        return all_preds
        

        
        


    




        



        
        

    
    
    

        

Criterion = {
    'rank':RankingCriterion,
    'none':XEntCriterion,
    'none_margin':MarginCriterion,
    'linearxent':LinearXEntCriterion,
    'mean': MeanCriterion
}




        
        

    
    
    



   
