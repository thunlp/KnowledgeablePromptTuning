export CUDA_VISIBLE_DEVICES=2
mypred_methods=(none rank)
myoptimize_methods=(none rank)
mylabel_files=(label_names label_names_kb)



for mydataset in agnews
do

    for mymargin in 10
        do 
        for myweight_learning_rate in  0.03 
        do 
        for mypred_temperature in 3
        do 
        for template_id in 0
        do
            for myseed in  144
            do
            for config_id in 0
do  
            
                # cd  src && 
                # git add . &&
                # git commit -m "generated files on `date +'%Y-%m-%d %H:%M:%S'`" ;
                # git checkout  29a969f356b7315211b19bd00d46b10ee33e4a56  && sleep 1 && cd ../ &&
                python -u src/run_withprior.py \
                --data_dir datasets/ \
                --dataset $mydataset \
                --model_type roberta \
                --model_name_or_path "../plm_model_cache/roberta-large" \
                --output_dir_base outputlogs/fs_KPT/ \
                --per_gpu_train_batch_size 2 \
                --gradient_accumulation_steps 1 \
                --max_seq_length 512 \
                --warmup_steps 500 \
                --learning_rate 3e-5 \
                --learning_rate_for_new_token 5e-4  \
                --weight_decay 1e-2 \
                --adam_epsilon 1e-8 \
                --seed $myseed \
                --max_epochs 5 \
                --pred_method ${mypred_methods[config_id]} \
                --optimize_method ${myoptimize_methods[config_id]} \
                --label_word_file ${mylabel_files[config_id]} \
                --tuning 1 \
                --topkratio -1 \
                --autotopk 1 \
                --cut_off 1 \
                --div_prior 0 \
                --template_id $template_id \
                --num_examples_per_label 10 \
                --task train_and_save_few_shot_model \
                --pred_temperature $mypred_temperature \
                --margin $mymargin \
                --multitoken 0 \
                --tf_idf_filtering 0 \
                --weight_learning_rate 0.03 \
                --filtering_using_prior 0 #& PIDPY=$! &
                # sleep 2 && cd src ; git checkout master && cd ../
                # wait $PIDPY
            done
        done
    done
done
done
done
done