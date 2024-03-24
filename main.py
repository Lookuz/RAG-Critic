import os 
import torch

# from FastChat.fastchat.model.model_adapter import load_model
from transformers import GenerationConfig

from utils.utils import *
from utils.const import *
from utils.prompt import get_prompt_from_task
from datasets.datasets import bootstrap_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

if __name__ == "__main__":
    args = parse_args()

    # Load model and tokenizer
    # model, tokenizer = load_model(
    #     model_path=args.model_path,
    #     device=args.device, 
    #     num_gpus=args.num_gpus,
    #     max_gpu_memory=args.max_gpu_memory,
    #     load_8bit=args.load_8bit,
    #     cpu_offloading=args.cpu_offloading
    # )
    # cache_dir="/home/users/nus/e1101650/scratch/vicuna-7b-cache/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d"
    cache_dir="/home/users/nus/e1101650/scratch/vicuna-13b-16k-cache/models--lmsys--vicuna-13b-v1.5-16k/snapshots/17c61f9ca19f5a7a04e96b2cc0d9bcf2920cb8c2"

    tokenizer = AutoTokenizer.from_pretrained(cache_dir)

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cache_dir,
        quantization_config=bnb_config,
        device_map="cuda:0",
        local_files_only = True
    )

    # Load generation config
    generation_config = GenerationConfig(
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        do_sample=False
    )
    
    dataset_args = {
        "evidence_path" : args.evidence_path,
        "top_k" : args.evidence_top_k
    }

    # Load task-specific prompt template
    prompt = get_prompt_from_task(task=args.task)

    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path))

    if args.task == BOOTSTRAP_INCORRECT_RESPONSE_TASK or args.task == BOOTSTRAP_EVALUATION_GENERATION_TASK:
        with torch.no_grad():
            bootstrap_dataset(
                prompt, 
                ideal_number_tokens=args.ideal_number_tokens,
                dataset=args.dataset,
                data_path=args.data_path,
                dataset_args=dataset_args,
                model=model, tokenizer=tokenizer,
                generation_config=generation_config,
                batch_size=args.batch_size, num_workers=args.num_workers,
                save_path=args.save_path,
            )
    elif args.task == FINETUNE_CRITIC_TASK:
        pass
    else:
        raise AssertionError(f"Task {args.task} invalid!")
