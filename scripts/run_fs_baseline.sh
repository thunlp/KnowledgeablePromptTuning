export CUDA_VISIBLE_DEVICES=5
mypred_methods=(none)
myoptimize_methods=(none)
mylabel_files=(label_names)



for mydataset in agnews
do
    for config_id in 0
    do 
        for template_id in 0
        do
            for myseed in 144
            do
                python -u src/run_withprior.py \
                --data_dir datasets/ \
                --dataset $mydataset \
                --model_type roberta \
                --model_name_or_path "../plm_model_cache/roberta-large" \
                --output_dir_base outputlogs_xent_baselines \
                --per_gpu_train_batch_size 2 \
                --gradient_accumulation_steps 1 \
                --max_seq_length 512 \
                --warmup_steps 500 \
                --learning_rate 3e-5 \
                --learning_rate_for_new_token 5e-4 \
                --weight_decay 1e-2 \
                --adam_epsilon 1e-8 \
                --seed $myseed \
                --max_epochs 5 \
                --pred_method ${mypred_methods[config_id]} \
                --optimize_method ${myoptimize_methods[config_id]} \
                --label_word_file ${mylabel_files[config_id]} \
                --tuning 1 \
                --topkratio 0.1 \
                --cut_off 1.0 \
                --div_prior 0 \
                --template_id $template_id \
                --num_examples_per_label 10 \
                --task train_and_save_few_shot_model \
                --margin 10
            done
        done
    done
done