PYTHONPATH=python3
BASEPATH="./"
DATASET=yahoo #agnews dbpedia imdb amazon yahoo
TEMPLATEID=0 # 1 2 3
SEED=144 # 145 146 147 148
SHOT=5 # 0 1 10 20
VERBALIZER=kpt #
CALIBRATION="--calibration" # ""
FILTER=tfidf_filter # none
MODEL_NAME_OR_PATH="../plm_cache/roberta-large"
RESULTPATH="results_zeroshot"
OPENPROMPTPATH="OpenPrompt"

cd $BASEPATH

CUDA_VISIBLE_DEVICES=0 $PYTHONPATH zeroshot.py \
--model_name_or_path $MODEL_NAME_OR_PATH \
--result_file $RESULTPATH \
--openprompt_path $OPENPROMPTPATH \
--dataset $DATASET \
--template_id $TEMPLATEID \
--seed $SEED \
--verbalizer $VERBALIZER $CALIBRATION \
--filter $FILTER