#!/bin/bash

# Define the list of values for distributionshift
shifts=("distributionshift" "binary" "exponential" "its" "all")

# Loop over each value in the list
for shift in "${shifts[@]}"
do
  echo "Running with Dataset: $shift"

  # Run the accelerate launch command with the current shift value
  accelerate launch scripts/train_adaptive_paraphraser.py \
    --dataset_name="/home/a2diaa/repos/MarkMyWords/dpo_dataset/$shift/" \
    --model_name_or_path='meta-llama/Llama-2-7b-chat-hf' \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-4 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --eval_steps 50 \
    --output_dir="dpo_llama_$shift" \
    --warmup_steps 50 \
    --report_to wandb \
    --run_name "$shift/dpo_llama" \
    --num_train_epochs 1 \
    --seed 42 \
    --overwrite_output_dir \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=32 \
    --lora_alpha=16 \
    --gradient_checkpointing=true \
    --max_length=3586 \
    --max_prompt_length=1536 \
    --bf16 \
    --eval_strategy=steps \
    --save_strategy=steps \
    --save_steps=50

  echo "Finished running with Dataset: $shift"
done