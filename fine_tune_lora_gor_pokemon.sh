torchrun --standalone --nnodes=1 --nproc_per_node=2 train_text_to_image_lora.py \
      --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
      --dataset_name=lambdalabs/pokemon-blip-captions \
      --dataloader_num_workers=8 \
      --mixed_precision="fp16" \
      --resolution=512 --center_crop --random_flip \
      --train_batch_size=2 \
      --gradient_accumulation_steps=1 \
      --max_train_steps=15000 \
      --learning_rate=1e-04 \
      --max_grad_norm=1 \
      --lr_scheduler="cosine" --lr_warmup_steps=0 \
      --output_dir="sd-pokemon-model-lora" \
      --checkpointing_steps=5000 \
      --validation_prompt="cute dragon creature" \
      --seed=42 \
      --str_filters=up_blocks.*_lora\.up --num_groups=32 --reg_type=inter --ortho_decay=1e-6