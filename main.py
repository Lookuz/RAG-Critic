import os 
import torch

# from FastChat.fastchat.model.model_adapter import load_model
from transformers import GenerationConfig

from utils.utils import *
from utils.const import *
from utils.prompt import get_prompt_from_task
from triviaqa_datasets.datasets import bootstrap_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from finetune import finetune_with_triviaqa
from generate import generate_answers

if __name__ == "__main__":
    args,_ = parse_args()
    
    if args.task != FINETUNE_MODEL_TASK:
        generation_args,_ = parse_generation_args()

    elif args.task == FINETUNE_MODEL_TASK:
        print("Parsing finetuning arguments...")
        finetuning_args,_ = parse_finetuning_args()
        
    else:
        raise AssertionError(f"Task {args.task} invalid!")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=False,
    # )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        # quantization_config=bnb_config,
        device_map=args.device,
        # device_map="auto",
        local_files_only = True
    )

    # Load generation hyperparameters
    generation_config = GenerationConfig(
        temperature=generation_args.temperature,
        repetition_penalty=generation_args.repetition_penalty,
        max_new_tokens=generation_args.max_new_tokens,
        num_beams=generation_args.num_beams,
        num_return_sequences=generation_args.num_return_sequences,
        do_sample=False
    )
    
    dataset_args = {
        "evidence_path" : generation_args.evidence_path,
        "top_k" : generation_args.evidence_top_k
    }

    # Load task-specific prompt template
    prompt = get_prompt_from_task(task=args.task)

    if not os.path.exists(os.path.dirname(generation_args.save_path)):
        os.makedirs(os.path.dirname(generation_args.save_path))

    # Bootstrapping
    if args.task in [BOOTSTRAP_EVALUATION_GENERATION_TASK, BOOTSTRAP_INCORRECT_RESPONSE_TASK]:

        with torch.no_grad():
            bootstrap_dataset(
                args.task, prompt, 
                ideal_number_tokens=generation_args.ideal_number_tokens,
                dataset=args.dataset,
                data_path=args.data_path,
                dataset_args=dataset_args,
                model=model, tokenizer=tokenizer,
                generation_config=generation_config,
                batch_size=args.batch_size, num_workers=args.num_workers,
                save_path=generation_args.save_path,
            )

    # Finetuning
    elif args.task == FINETUNE_MODEL_TASK:
        finetune_with_triviaqa(
            model=model, tokenizer=tokenizer, dataset=args.dataset, data_path=args.data_path, output_data_dir=finetuning_args.save_data_path, output_model_dir=finetuning_args.save_model_path, batch_size=args.batch_size, args=finetuning_args)
    
    # Answer generation
    elif args.task in [GENERATE_RESPONSES_WITH_CRITIC, GENERATE_REPONSES]:

        if args.task == GENERATE_RESPONSES_WITH_CRITIC:
            # TODO: Add critic model loading
            pass

        generate_answers(
            args.task, prompt, 
            ideal_number_tokens=generation_args.ideal_number_tokens,
            dataset=args.dataset,
            data_path=args.data_path,
            dataset_args=dataset_args,
            model=model, tokenizer=tokenizer,
            generation_config=generation_config,
            batch_size=args.batch_size, num_workers=args.num_workers,
            save_path=generation_args.save_path,
        )

    else:
        raise AssertionError(f"Task {args.task} invalid!")
