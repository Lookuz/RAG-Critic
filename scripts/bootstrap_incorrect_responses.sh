# Compute parameters
device="cuda"
batch_size=1
num_workers=8
# max_gpu_memory="8GiB"
num_gpus=1

# Dataset parameters
dataset="triviaqa"
data_path="datasets/TriviaQA/rc/qa/wikipedia-train.json"
save_path="datasets/TriviaQA/rc/qa/bootstrap/web-train-incorrect-response.json"
evidence_path="datasets/TriviaQA/rc/evidence/wikipedia"
evidence_top_k=3

# Generation parameters
temperature=0.
repetition_penalty=1
max_new_tokens=512
num_beams=1
num_return_sequences=1

# Latent Semantic Analysis parameters
ideal_number_tokens=8000

CUDA_VISIBLE_DEVICES=0 python main.py \
    --task "bootstrap-incorrect-response" \
    --num-gpus $num_gpus \
    --device $device \
    --dataset $dataset \
    --data-path $data_path \
    --save-path $save_path \
    --evidence-path $evidence_path \
    --evidence-top-k $evidence_top_k \
    --temperature $temperature \
    --repetition-penalty $repetition_penalty \
    --max-new-tokens $max_new_tokens \
    --num-beams $num_beams \
    --num-return-sequences $num_return_sequences \
    --ideal-number-tokens $ideal_number_tokens \
    --batch-size $batch_size \
    --load-8bit \
    
