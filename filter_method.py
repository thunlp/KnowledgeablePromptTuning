import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 



def tfidf_filter(myverbalizer, cc_logits, class_labels):
    myrecord = ""
    class_num = len(class_labels)
    norm_ord = 10/(class_num-2+1e-2) +1 
    print("norm_ord", norm_ord)
    context_size = cc_logits.shape[0]
    tobeproject = cc_logits.transpose(0,1).unsqueeze(0)
    ret = []
    for i in range(tobeproject.shape[-1]):
        ret.append(myverbalizer.project(tobeproject[:,:,i]).unsqueeze(-1))
    ret = torch.cat(ret, dim=-1)
    label_words_cc_logits = ret.squeeze()

    label_words_cc_logits = label_words_cc_logits - label_words_cc_logits.mean(dim=-1,keepdims=True)#, dim=-1)
 
    first_label_logits = label_words_cc_logits[:,0,:]
    orgshape = label_words_cc_logits.shape
    label_words_cc_logits = label_words_cc_logits.reshape(-1,context_size)
    sim_mat =  cosine_similarity(label_words_cc_logits.cpu().numpy(),first_label_logits.cpu().numpy() ).reshape(*orgshape[:-1],first_label_logits.shape[0])
    sim_mat = sim_mat - 10000.0* (1-myverbalizer.label_words_mask.unsqueeze(-1).cpu().numpy())

    new_label_words = []
    max_lbw_num_pclass = myverbalizer.label_words_mask.shape[-1]
    outputers = []
    for class_id in range(len(myverbalizer.label_words)):
        tfidf_scores = []
        tf_scores = []
        idf_scores = []
        num_words_in_class = len(myverbalizer.label_words[class_id])
        for in_class_id in range(max_lbw_num_pclass):
            if myverbalizer.label_words_mask[class_id, in_class_id] > 0:
                word_sim_scores = sim_mat[class_id, in_class_id]
                tf_score = word_sim_scores[class_id]
                idf_score_source = np.concatenate([word_sim_scores[:class_id], word_sim_scores[class_id+1:]])
                idf_score = 1/ (np.linalg.norm(idf_score_source, ord=norm_ord)/np.power((class_num-1), 1/norm_ord))
                tfidf_score = tf_score * idf_score #+1e-15)
                if tf_score<0:
                    tfidf_score = -100
                tfidf_scores.append(tfidf_score)
                tf_scores.append(tf_score)
                idf_scores.append(idf_score)
    
        outputer = list(zip(myverbalizer.label_words[class_id], 
                                            tfidf_scores,
                                            tf_scores,
                                            idf_scores))
        
        outputer = sorted(outputer, key=lambda x:-x[1])
        outputers.append(outputer)

    cut_optimality = []
    max_outputer_len = max([len(outputers[class_id]) for class_id in range(len(outputers))])
    for cut_potent in range(max_outputer_len):
        cut_rate = cut_potent/max_outputer_len
        loss = 0
        for class_id in range(len(myverbalizer.label_words)):
            cut_potent_this_class = int(cut_rate*len(outputers[class_id]))
            if len(outputers[class_id]) <= cut_potent_this_class:
                boundary_score = outputers[class_id][-1][1]
            else:
                boundary_score = outputers[class_id][cut_potent_this_class][1]
            loss += (boundary_score-1)**2
        cut_optimality.append([cut_rate, loss])
    optimal_cut_rate = sorted(cut_optimality, key=lambda x:x[1])[0][0]
    print("optimal_cut rate is {}".format(optimal_cut_rate))
    for class_id in range(len(myverbalizer.label_words)):
        cut = int(len(outputers[class_id])*optimal_cut_rate)
        if cut==0:
            cut=1
        # cut = optimal_cut
        new_l = [x[0] for x in outputers[class_id][:cut]]
        removed_words = [x[0] for x in outputers[class_id][cut:]]
        myrecord += f"Class {class_id} {new_l}\n"
        myrecord +=f"Class {class_id} rm: {removed_words}\n"
        new_label_words.append(new_l)
    myverbalizer.label_words = new_label_words
    myverbalizer = myverbalizer.cuda()
    noww_label_words_num = [len(myverbalizer.label_words[i]) for i in range(len(class_labels))]
    myrecord += f"Phase 3 {noww_label_words_num}\n"
    return myrecord


