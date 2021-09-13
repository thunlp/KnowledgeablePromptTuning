export CUDA_VISIBLE_DEVICES=3
mypred_methods=(mean)
myoptimize_methods=(mean)
mylabel_files=(label_names_kb)

config_id=0


for mydataset in dbpedia
do
        for template_id in 1
        do
            for myseed in 144
            do
                # cd  src && 
                # git add . &&
                # git commit -m "generated files on `date +'%Y-%m-%d %H:%M:%S'`" ;
                # git checkout  421ed06c86bc6e2d30a3ac0252b01833e65d1f63  && sleep 0.5 && cd ../ &&
                python -u src/run_withprior.py \
                --data_dir datasets/ \
                --dataset $mydataset \
                --model_type roberta \
                --model_name_or_path "../plm_model_cache/roberta-large" \
                --output_dir_base outputlogs/zs_KPT_domainpmi_$mydataset/ \
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
                --tuning 0 \
                --topkratio 0 \
                --autotopk 1 \
                --cut_off 0.5 \
                --div_prior 1 \
                --template_id $template_id \
                --num_examples_total 200 \
                --task run_zero_shot_model \
                --multitoken 1 \
                --tf_idf_filtering 0 \
                --filtering_using_prior 1 \
                --domainpmi 1
                #& PIDPY=$! &
                # sleep 1 && cd src ; git checkout master && cd ../
                # wait $PIDPY
                # --weight_learning_rate $myweight_learning_rate \
                # --pred_temperature $mypred_temperature \
                # --margin $mymargin
            done
        done
    done
