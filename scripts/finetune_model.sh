# Compute parameters
device="cuda"
batch_size=4
num_workers=8

# Dataset parameters
dataset="triviaqa"
# model_path="/home/users/nus/e1101650/scratch/vicuna-13b-16k-cache/models--lmsys--vicuna-13b-v1.5-16k/snapshots/17c61f9ca19f5a7a04e96b2cc0d9bcf2920cb8c2"
model_path="/home/users/nus/e1101650/scratch/llama2-chat-hf-cache/models--meta-llama--Llama-2-7b-chat-hf/snapshots/09bd0f49e16738cdfaa6e615203e126038736eb0"
data_path="triviaqa_datasets/TriviaQA/rc/qa/wikipedia-train.json"
save_data_path=triviaqa_datasets
save_model_path="finetuned_models/base_model"

# PEFT parameters
lora_alpha=32
lora_dropout=0.1
r=16
bias="none"
task_type="CAUSAL_LM"

# SFTT parameters
num_train_epochs=1
gradient_accumulation_steps=2
optim="paged_adamw_32bit"
logging_steps=10
save_strategy="steps"
evaluation_strategy="steps"
eval_steps=10
do_eval=True
learning_rate=2e-4
max_grad_norm=0.3
warmup_ratio=0.03
lr_scheduler_type="constant"


CUDA_VISIBLE_DEVICES=0 python main.py \
    --task "finetune_model" \
    --model_path $model_path \
    --device $device \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --dataset $dataset \
    --data_path $data_path \
    --save_data_path $save_data_path \
    --save_model_path $save_model_path \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --r $r \
    --bias $bias \
    --task_type $task_type \
    --batch_size $batch_size \
    --num_train_epochs $num_train_epochs \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --gradient_checkpointing \
    --optim $optim \
    --logging_steps $logging_steps \
    --save_strategy $save_strategy \
    --learning_rate $learning_rate \
    --max_grad_norm $max_grad_norm \
    --warmup_ratio $warmup_ratio \
    --lr_scheduler_type $lr_scheduler_type
