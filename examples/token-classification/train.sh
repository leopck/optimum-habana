export PYTHON=/usr/bin/python3.10
export PT_HPU_LAZY_MODE=0
export PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1
# $PYTHON train.py
rm -rf results
$PYTHON train.py \
  --model_name_or_path SCUT-DLVCLab/lilt-roberta-en-base \
  --gaudi_config_name Habana/roberta-base \
  --dataset_name nielsr/funsd-layoutlmv3 \
  --do_train \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --learning_rate 5e-5 \
  --max_steps 2500 \
  --output_dir ./results \
  --use_habana \
  --use_lazy_mode \
  --bf16 \
  --logging_steps 200 \
  --evaluation_strategy steps \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 2 \
  --load_best_model_at_end \
  --metric_for_best_model overall_f1 \
  --gradient_checkpointing \
  --use_hpu_graphs \
  --dataloader_num_workers 4 \
  --non_blocking_data_copy
