# Compute parameters
device="cuda"
batch_size=4
num_workers=8
# num_gpus=1
model_path="lmsys/vicuna-13b-v1.5"

# Dataset parameters
dataset="triviaqa"
data_path="triviaqa_datasets/TriviaQA/rc/qa/bootstrap/web_train_incorrect_response.json"
save_path="triviaqa_datasets/TriviaQA/rc/qa/bootstrap/web_train_evaluation_generation.json"
evidence_path="triviaqa_datasets/TriviaQA/rc/evidence/wikipedia"
evidence_top_k=3

# Generation parameters
temperature=0.
repetition_penalty=1
max_new_tokens=512
num_beams=1
num_return_sequences=1

CUDA_VISIBLE_DEVICES=0 python main.py \
    --task "bootstrap-evaluation-generation" \
    --model_path $model_path \
    --device $device \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --dataset $dataset \
    --data_path $data_path \
    --save_path $save_path \
    --evidence_path $evidence_path \
    --evidence_top_k $evidence_top_k \
    --temperature $temperature \
    --repetition_penalty $repetition_penalty \
    --max_new_tokens $max_new_tokens \
    --num_beams $num_beams \
    --num_return_sequences $num_return_sequences \
    --do-sample \
    --load-8bit
