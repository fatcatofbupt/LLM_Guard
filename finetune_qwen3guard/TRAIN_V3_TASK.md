# Training Task: Fine-tune Qwen3Guard with Augmented Dataset v3

## Objective
Retrain the safety model using the augmented dataset v3 (train_v3.jsonl + val_v3.jsonl)
to reduce false negatives on the 拒答 sheet.

## Prerequisites
- Conda environment `py311` with unsloth, transformers, trl, datasets, accelerate
- Base model at `models/Qwen/Qwen3Guard-Gen-0.6B`
- Augmented dataset at:
  - `finetune_qwen3guard/data/train_v3.jsonl` (4,019 records)
  - `finetune_qwen3guard/data/val_v3.jsonl` (300 records)

## Command

```bash
conda run --no-capture-output -n py311 python3 finetune_qwen3guard/scripts/02_train_unsloth.py \
  --train_file finetune_qwen3guard/data/train_v3.jsonl \
  --val_file finetune_qwen3guard/data/val_v3.jsonl \
  --output_dir finetune_qwen3guard/output/unsloth_lora_v3 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --save_merged
```

## Hyperparameters
| Param | Value |
|-------|-------|
| Epochs | 3 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Effective batch | 16 |
| Learning rate | 2e-4 |
| LoRA r | 16 |
| LoRA alpha | 32 |
| Max seq length | 2048 |
| Seed | 42 |

## Output
- LoRA adapter: `finetune_qwen3guard/output/unsloth_lora_v3/`
- Merged model: `finetune_qwen3guard/output/unsloth_lora_v3/merged_model/`

## Post-Training Update
After training, update these files to use the new model path:

1. `safety_service.py`:
   ```python
   SAFETY_MODEL_PATH = "./finetune_qwen3guard/output/unsloth_lora_v3/merged_model"
   ```

2. `batch_process_two_stage.py`:
   ```python
   SAFETY_MODEL_PATH = "./finetune_qwen3guard/output/unsloth_lora_v3/merged_model"
   ```

3. `hello_qwen3guard.py` (if applicable)

## Estimated Time
~30-60 minutes on RTX 4090 (CUDA)

## Dataset v3 Stats
- Total train: 4,019 (was 2,315 in v2)
- Safe: 905
- Unsafe: 3,110
- Added from false negatives: ~1,700
- Key additions: Unethical Acts (+1,600), Copyright (+203), Illegal (+90), PII (+45), Violent (+36), Political (+30)

## Notes
- The `--save_merged` flag merges LoRA weights into the base model automatically
- If OOM, reduce `--per_device_train_batch_size` to 2 or enable 8-bit: `--load_in_8bit`
- Training logs saved to `finetune_kill.log`
