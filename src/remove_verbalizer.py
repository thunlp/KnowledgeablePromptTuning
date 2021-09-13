## Not in use currently

import torch
import logger
log = logger.get_logger(__name__)


class Removewords(object):
    def __init__(self,prompt_label_idx ):
        self.prompt_label_idx = prompt_label_idx
        self.labelword_ave_loss = []

    def append(self, logits):
        self.labelword_ave_loss.append(logits)
    
    def remove_uninformative(self):
        log.debug("removing words!")
        tmp = torch.cat(self.labelword_ave_loss, dim=0)
        mean = torch.mean(tmp, dim=0)
        std = torch.std(tmp, dim=0)

        torch.set_printoptions(edgeitems=10)
        crit = mean+std
        to_remove = torch.argsort(crit)[:,:1].tolist()
        # torch.argsort()


        new_prompt_label_idx = []
        for i in range(len(self.prompt_label_idx)):
            new_prompt_label_idx.append([j for idx,j in enumerate(self.prompt_label_idx[i]) if idx not in to_remove[i]])
        new_prompt_label_idx = torch.tensor(new_prompt_label_idx).cuda()

        
        
        log.debug("{} {} {} {}".format(std, mean, new_prompt_label_idx.size(), self.labelword_ave_loss[0].size()))
        return new_prompt_label_idx
