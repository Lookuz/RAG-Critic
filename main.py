import os
import torch

from fastchat.model.model_adapter import load_model
from transformers import GenerationConfig

from utils import const
from utils.incorrect_response_gen import get_prompt_from_task
from datasets.datasets import bootstrap_dataset

if __name__ == "__main__":

    task = const.BOOTSTRAP_INCORRECT_RESPONSE_TASK
    device = "cuda"
    batch_size = 8
    num_workers = 16
    max_gpu_memory = "20GiB"
    num_gpus = 1
    # Dataset parameters
    dataset = "triviaqa"
    data_path = "datasets/TriviaQA/rc/qa/wikipedia-train.json"
    save_path = "datasets/TriviaQA/rc/qa/bootstrap/web-train-incorrect-response.json"
    evidence_path = "datasets/TriviaQA/rc/evidence/wikipedia"
    evidence_top_k = 3
    do_sample = False
    # Generation parameters
    temperature = 1.
    repetition_penalty = 1
    max_new_tokens = 100
    num_beams = 1
    num_return_sequences = 1
    load_8bit = False
    cpu_offloading = True

    # Load model and tokenizer
    model, tokenizer = load_model(
        model_path="lmsys/vicuna-7b-v1.5",
        device=device,
        num_gpus=num_gpus,
        max_gpu_memory=max_gpu_memory,
        load_8bit=load_8bit,
        cpu_offloading=cpu_offloading
    )
    model = model.to(device)

    # Load generation config
    generation_config = GenerationConfig(
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample
    )
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
            bootstrap_dataset(
                prompt,
                dataset=dataset,
                data_path=data_path,
                dataset_args=dataset_args,
                model=model, tokenizer=tokenizer,
                generation_config=generation_config,
                batch_size=batch_size, num_workers=num_workers,
                save_path=save_path
            )
    elif task == const.FINETUNE_CRITIC_TASK:
        pass
    else:
        raise AssertionError(f"Task {task} invalid!")
