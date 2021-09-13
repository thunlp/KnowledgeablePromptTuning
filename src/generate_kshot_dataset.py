
import argparse
import json
import numpy as np
from numpy.random import seed
import torch
import os 


def main(args, repetition):
    labelfile = open(os.path.join(args.dataset_path, args.dataset_name,'train_labels.txt'),'r')
    labels = [int(x.strip()) for x in labelfile.readlines()]
    samples_by_label = {}
    samples_by_label_dev = {}
    with open(os.path.join(args.dataset_path, args.dataset_name,'train.txt'),'r') as fin:
        lines = fin.readlines()
        order = np.arange(len(lines))
        np.random.shuffle(order)
        for idx in order:
            if labels[idx] not in samples_by_label:
                samples_by_label[labels[idx]] = [lines[idx].strip()]
            elif len(samples_by_label[labels[idx]])<args.shot:
                samples_by_label[labels[idx]].append(lines[idx].strip())
            elif labels[idx] not in samples_by_label_dev:
                samples_by_label_dev[labels[idx]] = [lines[idx].strip()]
            elif len(samples_by_label_dev[labels[idx]])<args.shot:
                samples_by_label_dev[labels[idx]].append(lines[idx].strip())
            else:
                continue
    
    write_file =  open(os.path.join(args.dataset_path, args.dataset_name,'train_{}-shot_rep-{}.json'.format(args.shot, repetition)),'w')
    dl = []
    for key in samples_by_label:
        for text in samples_by_label[key]:
            dl.append({'text':text, 'label':key})
    order = np.arange(len(dl))
    np.random.shuffle(order)
    for idx in order:
        write_file.write(str(dl[idx])+'\n')


    write_file =  open(os.path.join(args.dataset_path, args.dataset_name,'dev_{}-shot_rep-{}.json'.format(args.shot, repetition)),'w')
    dl = []
    for key in samples_by_label_dev:
        for text in samples_by_label_dev[key]:
            dl.append({'text':text, 'label':key})
    order = np.arange(len(dl))
    np.random.shuffle(order)
    for idx in order:
        write_file.write(str(dl[idx])+'\n')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--shot", type=int, default=16)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--repetition",type=int, default=5)
    args = parser.parse_args()

    np.random.seed(args.seed)
    for i in range(args.repetition):
        main(args = args, repetition=i)



            
            

