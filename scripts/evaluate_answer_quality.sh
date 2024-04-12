# Compute parameters
device="cuda"
batch_size=16
num_workers=8

# Dataset parameters
dataset="triviaqa"
model_path="meta-llama/Llama-2-7b-chat-hf"
# model_path="/home/users/nus/e1101650/scratch/llama2-chat-hf-cache/models--meta-llama--Llama-2-7b-chat-hf/snapshots/09bd0f49e16738cdfaa6e615203e126038736eb0"
data_path="triviaqa_datasets/TriviaQA/rc/qa/bootstrap/web-dev-refined-generated-response.json"
metric="GLEU"

save_data_path="triviaqa_datasets/answer_evaluation_results/{$metric}.json"
save_every=10

# Prompt the user for the authentication token
echo -n "Enter your HuggingFace authentication token for using LLaMa 2: "
read -s auth_token
echo

CUDA_VISIBLE_DEVICES=0 python main.py \
    --task "evaluate_answers_quality" \
    --metric $metric \
    --model_path $model_path \
    --device $device \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --dataset $dataset \
    --data_path $data_path \
    --hf_token $auth_token \
    --save_data_path $save_data_path \
    --save_every $save_every
