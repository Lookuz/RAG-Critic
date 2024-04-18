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
from generate import generate_answers, extract_answers
from eval_answer_quality import evaluate_answers_quality

if __name__ == "__main__":
    args,_ = parse_args()

    if args.task == FINETUNE_CRITIC_TASK:
        print("Parsing finetuning arguments...")
        finetuning_args, _ = parse_finetuning_args()

    elif args.task == EVALUATE_ANSWERS_QUALITY_TASK:
        print("Parsing evaluation arguments...")
        evaluation_args, _ = parse_evaluation_args()

    else:
        generation_args, _ = parse_generation_args()

    if args.task != EVALUATE_ANSWERS_QUALITY_TASK:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            # device_map=args.device,
            device_map="auto",
            local_files_only = True
        )

    # Bootstrapping
    if args.task in [BOOTSTRAP_EVALUATION_GENERATION_TASK, BOOTSTRAP_INCORRECT_RESPONSE_TASK, GENERATE_RESPONSES_TASK, POST_PROCESS_RESPONSE_TASK]:            
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

        # Standard generation for test dataset
        if args.task == GENERATE_RESPONSES_TASK:
            with torch.no_grad():
                generate_answers(
                    args.task, prompt, 
                    dataset=args.dataset,
                    data_path=args.data_path,
                    dataset_args=dataset_args,
                    model=model, tokenizer=tokenizer,
                    generation_config=generation_config,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    save_path=generation_args.save_path,
                )
        # Standard generation for test dataset
        elif args.task == POST_PROCESS_RESPONSE_TASK:
            with torch.no_grad():
                extract_answers(
                    args.task, prompt, 
                    dataset=args.dataset,
                    data_path=args.data_path,
                    dataset_args=dataset_args,
                    model=model, tokenizer=tokenizer,
                    generation_config=generation_config,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    save_path=generation_args.save_path,
                )                

        else:
            # Bootstrapping for incorrect responses / evaluations
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
    elif args.task == FINETUNE_CRITIC_TASK:
        finetune_with_triviaqa(
            model=model, tokenizer=tokenizer, dataset=args.dataset, data_path=args.data_path, output_data_dir=finetuning_args.save_data_path, output_model_dir=finetuning_args.save_model_path, batch_size=args.batch_size, args=finetuning_args)
    
    # Evaluation by critic
    elif args.task == REFINE_RESPONSE_WITH_CRITIC_TASK:
        # Load critic model and tokenizer
        print("Loading critic model...")
        critic_tokenizer = AutoTokenizer.from_pretrained(generation_args.critic_model_path)

        critic_model = AutoModelForCausalLM.from_pretrained(
            generation_args.critic_model_path,
            quantization_config=bnb_config,
            # device_map=args.device,
            device_map="auto",
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

        
        # Construct prompt for critic model and for rewriting of the original response using feedback
        critic_prompt = get_prompt_from_task(REFINE_RESPONSE_WITH_CRITIC_TASK)
        rewrite_prompt = get_prompt_from_task(RESPONSE_REWRITE_TASK)

        generate_answers(
            args.task, prompt, 
            dataset=args.dataset,
            data_path=args.data_path,
            dataset_args=dataset_args,
            model=model, tokenizer=tokenizer,
            critic_model=critic_model, critic_tokenizer=critic_tokenizer,
            critic_prompt=critic_prompt, rewrite_prompt=rewrite_prompt,
            generation_config=generation_config,
            batch_size=args.batch_size, num_workers=args.num_workers,
            save_path=generation_args.save_path,
        )

    # Evaluate answer for critic-refined vs zero-shot comparison
    elif args.task == EVALUATE_ANSWERS_QUALITY_TASK:
        if not os.path.exists(os.path.dirname(evaluation_args.save_path)):
            os.makedirs(os.path.dirname(evaluation_args.save_path))

        token = evaluation_args.hf_token if len(evaluation_args.hf_token) else None
        with torch.no_grad():
            evaluate_answers_quality(
                task=args.task,
                dataset=args.dataset,
                data_path=args.data_path,
                batch_size=args.batch_size,
                metric=evaluation_args.metric,
                save_path=evaluation_args.save_path,
                auth_token=token,
                eval_model_path=evaluation_args.eval_model_path,
                num_workers=args.num_workers,
                save_every=evaluation_args.save_every,
                device=args.device,
            )

    else:
        raise AssertionError(f"Task {args.task} invalid!")
