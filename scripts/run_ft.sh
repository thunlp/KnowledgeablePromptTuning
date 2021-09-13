export CUDA_VISIBLE_DEVICES=7
# mypred_methods=(none rank linearxent)
# myoptimize_methods=(none rank linearxent)
# mylabel_files=(label_names label_names_kb label_names_kb)



for mydataset in agnews imdb amazon dbpedia
do

            for myseed in   144 145 146 147 148
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
                --output_dir_base outputlogs/finetuning-20shot/ \
                --per_gpu_train_batch_size 2 \
                --gradient_accumulation_steps 1 \
                --max_seq_length 512 \
                --warmup_steps 500 \
                --learning_rate 3e-5 \
                --learning_rate_for_new_token 5e-4  \
                --weight_decay 1e-2 \
                --adam_epsilon 1e-8 \
                --seed $myseed \
                --label_word_file label_names \
                --max_epochs 5 \
                --template_id 0 \
                --num_examples_per_label 20 \
                --task run_finetuning \
                --learning_rate_for_classifier 1e-3 \
                --finetuning
            done
done