# TRL Reward Model Training — Simple Guide

This guide explains how to run reward model training with TRL and LoRA, including all the configs we have set to make it work.

## 1. Setup Scripts

### Linux / WSL / macOS
```bash
./setup.sh              # Run setup
```

```bash
# in oreder to set a safe folder for WandB logs to avoid errors
export WANDB_DIR=$PWD/wandb_temp
mkdir -p $WANDB_DIR
```

## 2 Config File

The main config file is:

```
examples/cli_configs/reward_lora_config.yaml
```
Or

```
examples/cli_configs/reward_lora_macbook_config.yaml
```

### Important parts to change:

* **Model path**:

  ```yaml
  model_name_or_path: "Qwen/Qwen2.5-0.5B-Instruct"
  ```

  Change this if you want a different model.

* **Dataset path**:

  ```yaml
  datasets:
    - path: "AmirMohseni/lmarena-preference-data-edu-en-first-round"
      split: "train"
  ```

  Change this to your desired dataset.

* **Field mapping**:

  ```yaml
  field_mapping:
    chosen_field: "chosen"
    rejected_field: "rejected"
  ```

  Make sure it matches your dataset columns.

* **Batch size / sequence length** (memory usage):

  ```yaml
  # I've set these to train a simple model with low memory usage, but you'll need to modify them based on your purpose
  per_device_train_batch_size: 1
  max_length: 256
  gradient_accumulation_steps: 4
  ```

* **LoRA settings**:

  ```yaml
  use_peft: true
  lora_r: 8 # set LoRA rank and tune it
  lora_alpha: 16
  lora_dropout: 0.01
  lora_target_modules: "all-linear"
  lora_task_type: "SEQ_CLS"
  ```

* **Windows multiprocessing fixes**:

  ```yaml
  dataset_num_proc: 1
  dataloader_num_workers: 0
  ```

## 3. Set HF_Token and WandB API key:
Open `.env.example` file, and fill the following variables with your own tokens:

```.env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## 4. Commands to run

### Train reward model
```bash
python examples/scripts/reward_trainer.py --config examples/cli_configs/reward_lora_config.yaml
```

## 5. Common issues and changes:

1. **Reduced memory usage**

   * Switched to smaller model `Qwen2.5-0.5B-Instruct`.
   * BF16 mixed precision + gradient checkpointing.
   * Small batch size + gradient accumulation.

2. **LoRA setup**

   * Added LoRA with rank 8, alpha 16, dropout 0.01.
   * Targeted linear layers only.

3. **Windows-safe multiprocessing**

   * `dataset_num_proc = 1`
   * `dataloader_num_workers = 0`

4. **WandB logging fixed**

   * Set `WANDB_DIR` to a temp folder to avoid `FileNotFoundError`.

5. **Quantization**

   * 8-bit quantization (`load_in_8bit: true`) or disabled quantization for Windows.

6. Evaluation & Saving

During training, the reward model performs step-based evaluation and automatically saves checkpoints. You can configure these behaviors in your YAML config file (examples/cli_configs/reward_lora_config.yaml).

```yml
# 1. Evaluation settings
eval_strategy: "steps"     # Can also be "epochs" if you prefer epoch-based evaluation
eval_steps: 25             # Evaluate the model every 25 training steps

# 2. Checkpoint saving settings
save_strategy: "steps"     # Can also be "epochs"
save_steps: 50             # Save a checkpoint every 50 steps
save_total_limit: 3        # Keep only the 3 most recent checkpoints

# 3. Output directories and logging
  # 1. Directory to save checkpoints, logs, and model outputs
  output_dir: "runs/reward-lmarena-preference-data-edu-en-first-round-LoRA-rank8"

  # 2. Name of this training run (used in WandB)
  run_name: "reward-lmarena-preference-data-edu-en-first-round-LoRA-rank8"

  # 3. Project name in WandB
  project_name: "reward-model-training"
``` 
## Summary

* Edit the **config file** for model, dataset, and batch size.
* Run **setup script** (`setup.sh`).
* Run **reward trainer** with `python examples/scripts/reward_trainer.py --config examples/cli_configs/reward_lora_config.yaml`.
