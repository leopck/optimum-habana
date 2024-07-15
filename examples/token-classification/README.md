# Getting Started

1. Install the required dependencies
```bash
apt update
apt install -y tesseract-ocr
pip install -r requirements.txt
```

2. Run Fine-tuning on FUND dataset for SCUT-DLVCLab/lilt-roberta-en-base

fine tune your own dataset:
```bash
python train.py \
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
```

3. Run inference based on Fined-tuned dataset

This assumes that the output_dir is located at `results` directory:
```
python infer.py
```

## Multi-device using DeepSpeed

```
python train.py \
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
  --logging_strategy epoch \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --save_steps 200 \
  --save_total_limit 2 \
  --load_best_model_at_end \
  --metric_for_best_model overall_f1 \
  --gradient_checkpointing \
  --use_hpu_graphs \
  --dataloader_num_workers 4 \
  --non_blocking_data_copy \
  --report_to tensorboard \
  --deepspeed ds_config_bf16.json
```