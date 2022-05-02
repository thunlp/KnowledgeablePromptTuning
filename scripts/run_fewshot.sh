PYTHONPATH=python
BASEPATH="./"
DATASET=amazon #agnews dbpedia imdb amazon yahoo
TEMPLATEID=0 # 1 2 3
SEED=144 # 145 146 147 148
SHOT=5 # 0 1 10 20
VERBALIZER=kpt #soft auto
FILTER=tfidf_filter # none
KPTWLR=0.0 # 0.06
MAXTOKENSPLIT=-1 # 1
MODEL_NAME_OR_PATH="../plm_cache/roberta-large"
RESULTPATH="results_fewshot"
OPENPROMPTPATH="OpenPrompt"

cd $BASEPATH

CUDA_VISIBLE_DEVICES=7 $PYTHONPATH fewshot.py \
--model_name_or_path $MODEL_NAME_OR_PATH \
--result_file $RESULTPATH \
--openprompt_path $OPENPROMPTPATH \
--result_file results_fewshot_norefine.txt \
--dataset $DATASET \
--template_id $TEMPLATEID \
--seed $SEED \
--shot $SHOT \
--verbalizer $VERBALIZER \
--max_token_split $MAXTOKENSPLIT \
--kptw_lr $KPTWLR