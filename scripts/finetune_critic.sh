# Compute parameters
device="cuda"
batch_size=16
num_workers=8

# Dataset parameters
dataset="triviaqa"
# model_path="meta-llama/Llama-2-7b-chat-hf"
model_path="/home/users/nus/e1101650/scratch/llama2-chat-hf-cache/models--meta-llama--Llama-2-7b-chat-hf/snapshots/09bd0f49e16738cdfaa6e615203e126038736eb0"
data_path="triviaqa_datasets/TriviaQA/rc/qa/bootstrap/web_train_evaluation_generation.json"
save_data_path="triviaqa_datasets"
save_model_path="finetuned_models"

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
save_steps=20
evaluation_strategy="steps"
eval_steps=20
do_eval=True
learning_rate=2e-4
max_grad_norm=0.3
warmup_ratio=0.03
lr_scheduler_type="constant"


CUDA_VISIBLE_DEVICES=0 python main.py \
    --task "finetune_critic" \
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
    --num_train_epochs $num_train_epochs \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --gradient_checkpointing \
    --optim $optim \
    --logging_steps $logging_steps \
    --save_strategy $save_strategy \
    --save_steps $save_steps \
    --evaluation_strategy $evaluation_strategy \
    --eval_steps $eval_steps \
    --do_eval $do_eval \
    --learning_rate $learning_rate \
    --max_grad_norm $max_grad_norm \
    --warmup_ratio $warmup_ratio \
    --lr_scheduler_type $lr_scheduler_type
