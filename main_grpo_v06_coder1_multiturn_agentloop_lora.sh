#!/bin/bash
# Additive coder1 migration script for current verl AgentLoop stack.
# This script is new-only and does not change legacy training entrypoints.
set -x

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$SCRIPT_DIR/verl:$SCRIPT_DIR:${PYTHONPATH}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

MAX_EPOCHS=${MAX_EPOCHS:-8}
DATASET=${DATASET:-/mnt/data/jiahualu1/CodeAgent2/code-r1/code-r1-12k}
MODEL_PATH=${MODEL_PATH:-/mnt/data/jiahualu/LLM/Qwen2.5-3B/Qwen2.5-3B-Instruct}

AGENT_LOOP_CONFIG_PATH=${AGENT_LOOP_CONFIG_PATH:-$SCRIPT_DIR/verl/recipe/coder1_agentloop/new_agent_loop_config.yaml}
REWARD_FN_PATH=${REWARD_FN_PATH:-$SCRIPT_DIR/verl/recipe/coder1_agentloop/new_code_agentloop_reward.py}

ROLLOUT_N_SAMPLE=${ROLLOUT_N_SAMPLE:-16}
ROLLOUT_N_QUERY=${ROLLOUT_N_QUERY:-32}
MICRO_BATCH_PER_GPU=${MICRO_BATCH_PER_GPU:-2}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-4}
GLOBAL_BATCH_SIZE=$(( (GPUS_PER_NODE * MICRO_BATCH_PER_GPU) * GRAD_ACC_STEPS ))

LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}

INFER_TP=${INFER_TP:-2}
MAX_TURNS=${MAX_TURNS:-4}

TOTAL_SAMPLES=$((ROLLOUT_N_QUERY * ROLLOUT_N_SAMPLE))
if (( TOTAL_SAMPLES % GLOBAL_BATCH_SIZE != 0 )); then
    echo "Error: (ROLLOUT_N_QUERY * ROLLOUT_N_SAMPLE) must be divisible by GLOBAL_BATCH_SIZE."
    echo "Currently, ${TOTAL_SAMPLES} is not divisible by ${GLOBAL_BATCH_SIZE}."
    exit 1
fi

export VLLM_USE_V1=1
export RAY_DISABLE_DASHBOARD=1
export RAY_DASHBOARD_ENABLED=0

python3 -m verl.trainer.new_main_ppo_agentloop \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATASET/train.parquet \
    data.val_files=$DATASET/test.parquet \
    data.train_batch_size=$ROLLOUT_N_QUERY \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.model.target_modules=$LORA_TARGET_MODULES \
    actor_rollout_ref.model.use_shm=${USE_SHM:-False} \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-5e-6} \
    actor_rollout_ref.actor.ppo_mini_batch_size=$GLOBAL_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$INFER_TP \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=256 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL:-0.2} \
    actor_rollout_ref.rollout.n=$ROLLOUT_N_SAMPLE \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=${LAYERED_SUMMON:-True} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_TURNS \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_TURNS \
    actor_rollout_ref.rollout.agent.default_agent_loop=coder1_multiturn_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$AGENT_LOOP_CONFIG_PATH \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=256 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    custom_reward_function.path=$REWARD_FN_PATH \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='code-r1' \
    trainer.experiment_name=${DATASET}-grpo-v06-coder1-multiturn-agentloop-lora \
    trainer.nnodes=1 \
    trainer.default_local_dir=./models/${DATASET}-grpo-v06-coder1-multiturn-agentloop-lora \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.save_freq=64 \
    trainer.test_freq=16 \
    trainer.total_epochs=$MAX_EPOCHS \
    reward_model.reward_manager=prime "$@" 2>&1 | tee grpo_v06_coder1_multiturn_agentloop_lora.log
