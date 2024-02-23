# Compute parameters
device="mps"
batch_size=1
num_workers=4

# Dataset parameters
dataset="triviaqa"
data_path="datasets/TriviaQA/rc/qa/web-train.json"
save_path="datasets/TriviaQA/rc/qa/bootstrap/web-train-incorrect-response.json"
evidence_top_k=3

# Generation parameters
temperature=0.7
repetition_penalty=1
max_new_tokens=512
num_beams=1
num_return_sequences=1

CUDA_VISIBLE_DEVICES=0 python main.py \
    --task "bootstrap-incorrect-response" \
    --device $device \
    --dataset $dataset \
    --data-path $data_path \
    --save-path $save_path \
    --evidence-top-k $evidence_top_k \
    --temperature $temperature \
    --repetition-penalty $repetition_penalty \
    --max-new-tokens $max_new_tokens \
    --num-beams $num_beams \
    --num-return-sequences $num_return_sequences \
    --do-sample \
    --load-8bit
