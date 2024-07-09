export PYTHON=/usr/bin/python3.10
rm -rf results
$PYTHON train.py \
  --model_name_or_path SCUT-DLVCLab/lilt-roberta-en-base \
  --dataset_name nielsr/funsd-layoutlmv3 \
  --do_train \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --learning_rate 5e-5 \
  --max_steps 2500 \
  --output_dir ./results \
  --bf16 \
  --logging_steps 200 \
  --evaluation_strategy steps \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 2 \
  --load_best_model_at_end \
  --metric_for_best_model overall_f1 \
  --gradient_checkpointing \
  --dataloader_num_workers 4
