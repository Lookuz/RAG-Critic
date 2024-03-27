# Compute parameters
device="cuda"
batch_size=4
num_workers=8
# max_gpu_memory="8GiB"
# num_gpus=1

# Dataset parameters
dataset="triviaqa"
data_path="triviaqa_datasets/TriviaQA/rc/qa/wikipedia-train.json"
save_path="triviaqa_datasets/TriviaQA/rc/qa/bootstrap/web-train-incorrect-response.json"
evidence_path="triviaqa_datasets/TriviaQA/rc/evidence/wikipedia"
evidence_top_k=3

# Generation parameters
temperature=0.
repetition_penalty=1
max_new_tokens=512
num_beams=1
num_return_sequences=1

ideal_number_tokens=2000

CUDA_VISIBLE_DEVICES=0 python main.py \
    --task "bootstrap-incorrect-response" \
    --num_gpus $num_gpus \
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
    --ideal_number_tokens $ideal_number_tokens
