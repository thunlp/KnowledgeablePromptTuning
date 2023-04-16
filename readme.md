
# KPT source code

Here is the source code for our ACL 2022 paper 
[Knowledgeable Prompt-tuning: Incorporating Knowledge into Prompt Verbalizer for Text Classification](https://arxiv.org/abs/2108.02035)

## install openprompt 


Please install via git clone. This helps keep the dataset downloading scripts.

```bash
git clone git@github.com:thunlp/OpenPrompt.git
cd OpenPrompt
pip install -r requirements.txt
python setup.py install
```


## Download the dataset
```
cd OpenPrompt/datasets
bash download_text_classification.sh
```

## Run the scripts
for fewshot experiment
```
bash scripts/run_fewshot.sh 
```
for zeroshot experiment
```
bash scripts/run_zeroshot.sh
```
for pilot experiment in appendix
```
bash scripts/run_pilot.sh
```

The possible arguments in the scripts are in the comment of the scripts. 
Please choose the combination according to your need.
