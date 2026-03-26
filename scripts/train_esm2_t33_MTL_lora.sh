#!/bin/bash

# MTL LoRA ESM2-T33 training for grouped datasets
# Each group trains a separate MTL model

# Model configuration
plm_model_name=esm2_t33_650M_UR50D
plm_model_path="./models/esm2/esm2_t33_650M_UR50D"

# Optimized hyperparameters
lr=1.5e-4
training_method=mtl-lora

# Data root directory
data_group_dir="./dataset/data_group"

# Set working directory
if [ ! -d "$data_group_dir" ]; then
    echo "Error: Data group directory not found: $data_group_dir"
    exit 1
fi

# Groups to train
groups=("nucleotide" "inorganic_ion" "cofactor")

echo "Training MTL LoRA models for each group separately..."

# Loop through each group
for group_name in "${groups[@]}"; do
    echo "=========================================="
    echo "Training MTL LoRA model for group: $group_name"
    echo "=========================================="
    
    # Run training with optimized parameters for this group
    python trainer/train.py \
        --plm_model "$plm_model_path" \
        --training_method "$training_method" \
        --data_group_dir "$data_group_dir" \
        --mtl_group "$group_name" \
        --output_dir "./results/MTL_esm2_t33_lora_${group_name}" \
        --num_epochs 100 \
        --learning_rate "$lr" \
        --batch_size 8 \
        --gradient_accumulation_steps 4 \
        --warmup_steps 400 \
        --max_grad_norm 1.0 \
        --patience 8 \
        --min_delta 1e-3 \
        --min_epochs 5 \
        --monitor "aupr" \
        --monitor_strategy "max" \
        --num_workers 4 \
        --num_labels 2 \
        --problem_type "single_label_classification" \
        --metrics "mcc" "auroc" "aupr" \
        --lora_r 8 \
        --lora_alpha 24 \
        --lora_dropout 0.1 \
        --lora_target_modules "query" "key" "value" "dense"\
        --output_model_name "best_model.pt" \
        --max_seq_len 512
    
    if [ $? -eq 0 ]; then
        echo "MTL LoRA training completed successfully for group: $group_name"
    else
        echo "MTL LoRA training failed for group: $group_name"
    fi
    
    echo ""
done

echo "All MTL LoRA training completed!"
