
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=Debugging-Seq2Seq-PPO

accelerate launch ./examples/scripts/ppo/ppo_seq2seq.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --output_dir models/minimal/ppo \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --model_name_or_path google-t5/t5-small \
    --sft_model_path google-t5/t5-small \
    --reward_model_path google-t5/t5-small \
    --local_rollout_forward_batch_size 1 \
    --torch-dtype bfloat16 \
    --bf16 \
    --num-sample-generations 4 \
    --ds3-gather-for-generation False \
    --do-train \
    --missing_eos_penalty 1.0 \
    --report_to wandb
