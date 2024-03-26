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
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer = None

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_path,
    #     quantization_config=bnb_config,
    #     device_map=args.device,
    #     local_files_only = True
    # )
    model = None

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
                args.task, prompt, 
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
