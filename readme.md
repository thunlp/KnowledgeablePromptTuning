## Knowledgeable Prompt-tuning: Incorporating Knowledge into Prompt Verbalizer for Text Classification 

This is Knowledgeable Prompt-tuning: Incorporating Knowledge into Prompt Verbalizer for Text Classification (ACL 2022) 's source code ([Paper](https://arxiv.org/abs/2108.02035))


### Environments
```
python 3.7.10
torch==1.9.0
transformers==4.10.0
tqdm==4.48.1
numpy==1.18.5
sklearn==0.24.2
```

### Zero-shot text classification
```
bash scripts/run_zs_KPT_mean.sh
```

### Few-shot text classification
```
bash scripts/run_fs_KPT.sh
```

### Comments
Other scripts are for different experiments (including ablation study). Please refer to the paper for details. 
The impletation includes some unnecessary and dirty codes in prior experiments. I will clean it in the future or release a new version together with our future work. 
Due to some change in the version of package, the replicated results may differ slightly with results in the paper, but general trend is preversed.


### Link to originial experiment record
If you are interested, you can comment on the doc. (But sincerely speaking, I can't remember all the meaning of the numbers.)
https://docs.google.com/spreadsheets/d/124SaGGElKGv9Spdn05tDj5rKn9-gmiv8B_MEQqsMfXU/edit?usp=sharing

### Citation
```
@article{hu2021knowledgeable,
  title={Knowledgeable prompt-tuning: Incorporating knowledge into prompt verbalizer for text classification},
  author={Hu, Shengding and Ding, Ning and Wang, Huadong and Liu, Zhiyuan and Li, Juanzi and Sun, Maosong},
  journal={arXiv preprint arXiv:2108.02035},
  year={2021}
}
```
