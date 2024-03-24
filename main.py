import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import const
from utils.incorrect_response_gen import get_prompt_from_task
from datasets.datasets import bootstrap_dataset_mistral

if __name__ == "__main__":

    task = const.BOOTSTRAP_INCORRECT_RESPONSE_TASK
    device = "cuda"
    batch_size = 4
    num_workers = 16
    max_gpu_memory = "20GiB"
    num_gpus = 1
    # Dataset parameters
    dataset = "triviaqa"
    data_path = "datasets/TriviaQA/rc/qa/wikipedia-train.json"
    save_path = "datasets/TriviaQA/rc/qa/bootstrap/web-train-incorrect-response.json"
    evidence_path = "datasets/TriviaQA/rc/evidence/wikipedia"
    evidence_top_k = 3
    do_sample = True
    # Generation parameters
    temperature = 0.1
    repetition_penalty = 1
    max_new_tokens = 100
    num_beams = 1
    num_return_sequences = 1
    load_8bit = False
    cpu_offloading = False

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    dataset_args = {
        "evidence_path": evidence_path,
        "top_k": evidence_top_k
    }

    # Load task-specific prompt template
    prompt = get_prompt_from_task(task=task)

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    if task == const.BOOTSTRAP_INCORRECT_RESPONSE_TASK or task == const.BOOTSTRAP_WRONG_CONTEXT_TASK:
        with torch.no_grad():
            bootstrap_dataset_mistral(
                prompt,
                dataset=dataset,
                data_path=data_path,
                dataset_args=dataset_args,
                model=model, tokenizer=tokenizer,
                batch_size=batch_size, num_workers=num_workers,
                save_path=save_path
            )
    elif task == const.FINETUNE_CRITIC_TASK:
        pass
    else:
        raise AssertionError(f"Task {task} invalid!")
