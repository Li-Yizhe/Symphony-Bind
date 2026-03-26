#!/bin/bash

# Test MTL LoRA ESM2-T33 models for each group

# Model configuration
plm_model_name=esm2_t33_650M_UR50D
plm_model_path="./models/esm2/esm2_t33_650M_UR50D"
training_method=mtl-lora

# Data root directory
data_group_dir="./dataset/data_group"

# Set working directory
if [ ! -d "$data_group_dir" ]; then
    echo "Error: Data group directory not found: $data_group_dir"
    exit 1
fi

# Groups to test
groups=("nucleotide" "inorganic_ion" "cofactor")

echo "Testing MTL LoRA models for each group..."

# Loop through each group
for group_name in "${groups[@]}"; do
    echo "=========================================="
    echo "Testing MTL LoRA model for group: $group_name"
    echo "=========================================="
    
    model_dir="./results/MTL_esm2_t33_lora_${group_name}"
    
    # Check if model exists
    if [ ! -f "$model_dir/best_model.pt" ]; then
        echo "Warning: Model not found: $model_dir/best_model.pt"
        echo "Skipping group: $group_name"
        continue
    fi
    
    # Run testing
    python trainer/train.py \
        --plm_model "$plm_model_path" \
        --training_method "$training_method" \
        --data_group_dir "$data_group_dir" \
        --mtl_group "$group_name" \
        --output_dir "$model_dir" \
        --batch_size 8 \
        --num_workers 4 \
        --num_labels 2 \
        --problem_type "single_label_classification" \
        --metrics "mcc" "auroc" "aupr" \
        --max_seq_len 512 \
        --lora_r 8 \
        --lora_alpha 24 \
        --lora_dropout 0.1 \
        --lora_target_modules "query" "key" "value" "dense" \
        --output_model_name "best_model.pt" \
        --test_only
    
    if [ $? -eq 0 ]; then
        echo "Testing completed successfully for group: $group_name"
    else
        echo "Testing failed for group: $group_name"
    fi
    
    echo ""
done

echo "All MTL LoRA testing completed!"
