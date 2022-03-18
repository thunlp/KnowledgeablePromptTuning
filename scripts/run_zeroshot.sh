PYTHONPATH=/mnt/sfs_turbo/zhangshudan/anaconda3/envs/kpt/bin/python
BASEPATH=/mnt/sfs_turbo/hsd/thunlp_openprompt_private/
DATASET=agnews # dbpedia imdb amazon yahoo
TEMPLATEID=0 # 1 2 3
SEED=144 # 145 146 147 148
SHOT=5 # 0 1 10 20
VERBALIZER=kpt #
CALIBRATION="--calibration" # ""
FILTER=tfidf_filter # none
MODEL_NAME_OR_PATH="../plm_cache/roberta-large"
RESULTPATH="results_fewshot_softpilot"
OPENPROMPTPATH="OpenPrompt"

cd $BASEPATH

CUDA_VISIBLE_DEVICES=0 $PYTHONPATH forkptexp/zeroshot.py \
--model_name_or_path $MODEL_NAME_OR_PATH \
--result_file $RESULTPATH \
--openprompt_path $OPENPROMPTPATH \
--dataset $DATASET \
--template_id $TEMPLATEID \
--seed $SEED \
--verbalizer $VERBALIZER $CALIBRATION \
--filter $FILTER 