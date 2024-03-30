from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from triviaqa_datasets.datasets import ContextualizedQADatasetForCriticFinetuning
from utils.utils import convert_to_hf_dataset, plot_training_curve
from datasets import load_from_disk
import os
import json

def finetune_with_triviaqa(model, tokenizer, dataset, data_path, output_data_dir, output_model_dir, batch_size, args):
    print("Loading dataset...\n")
    hf_dataset_path = os.path.join(output_data_dir, "triviaqa_datasets_hf")

    if not os.path.exists(hf_dataset_path):
        triviaQA = ContextualizedQADatasetForCriticFinetuning.from_dataset(dataset=dataset, data_path=data_path)

        print("Convert to huggingface dataset...\n")
        triviaQA_hf = convert_to_hf_dataset(triviaQA)

        # Train-val split
        triviaQA_hf.save_to_disk(os.path.join(output_data_dir, "triviaqa_datasets_hf"))

        # Train val 90-10
        triviaQA_hf_split = triviaQA_hf.train_test_split(test_size=0.1)    
        triviaQA_hf_split["train"].save_to_disk(os.path.join(output_data_dir, "triviaqa_datasets_hf_train"))
        triviaQA_hf_split["test"].save_to_disk(os.path.join(output_data_dir, "triviaqa_datasets_hf_val"))

    dset_train = load_from_disk(os.path.join(output_data_dir, "triviaqa_datasets_hf_train"))
    dset_val = load_from_disk(os.path.join(output_data_dir, "triviaqa_datasets_hf_val"))
    print("Loaded dataset successfully!\n")

    print("Finetuning model...")

    # Finetune with PEFT
    model.config.pretraining_tp = 1
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.r,
        bias=args.bias,
        task_type=args.task_type,
    )
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=output_model_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        disable_tqdm=args.disable_tqdm
    )

    max_seq_length = 4096 # max sequence length for model 

    trainer = SFTTrainer(
        model=model,
        train_dataset=dset_train,
        eval_dataset=dset_val,
        peft_config=peft_config,
        tokenizer=tokenizer,
        packing=True,
        max_seq_length=max_seq_length,
        formatting_func=formatting_prompts_func,
        args=training_args,
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model()

    # Plot loss curve
    # Find the checkpoint folder
    subdirectories = [d for d in os.listdir(output_data_dir) if os.path.isdir(os.path.join(output_data_dir, d))]
    
    # Filter directories that start with "checkpoint"
    checkpoint_folder = [folder for folder in subdirectories if folder.startswith("checkpoint")][0]
    saved_states = os.path.join(output_data_dir,checkpoint_folder,"trainer_state.json")
    with open(saved_states, 'r') as file:
        # Load the JSON data
        data = json.load(file)
    log_history = data["log_history"]
    plot_training_curve(log_history)


# def generate_prompt(sample):
#     return f"""
#             You are an expert with general knowledge. 
#             ### Instruction: Based on your knowledge from Wikipedia articles, answer the question.
#             ### Input:
#             {question}

#             ### Response:
#             {answer}
#             """.format(question=sample["question"], answer=sample["answer"])

# Maybe can shift this to utils/const.py
def formatting_prompts_func(sample):
    return f"### Question: {sample['question']}\n ### Answer: {sample['answer']}"