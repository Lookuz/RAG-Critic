import argparse
import itertools
import random
from datasets import Dataset
import matplotlib.pyplot as plt

from utils.const import *

# Main arguments
def parse_args():
    parser = argparse.ArgumentParser()
    # Model 
    parser.add_argument(
        "--model_path",
        type=str,
        default="lmsys/vicuna-7b-v1.5",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "xpu", "npu"],
        default="cuda",
        help="The device type",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)

    # Task
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            BOOTSTRAP_INCORRECT_RESPONSE_TASK,
            BOOTSTRAP_EVALUATION_GENERATION_TASK,
            FINETUNE_CRITIC_TASK,
            REFINE_RESPONSE_WITH_CRITIC_TASK,
            GENERATE_RESPONSES_TASK,
            EVALUATE_ANSWERS_QUALITY_TASK,
            POST_PROCESS_RESPONSE_TASK
        ]
    )

    # Dataset and save paths
    parser.add_argument(
        "--dataset",
        type=str,
        default="triviaqa",
    )
    parser.add_argument("--data_path", type=str, required=True)
    return parser.parse_known_args()

# Generatation
def parse_generation_args():
    parser = argparse.ArgumentParser(description="Argument Parser for generation task")

    # Critic model
    parser.add_argument("--critic_model_path", type=str, default=None)

    # Generation 
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--do-sample", action='store_true')

    parser.add_argument("--evidence_path", type=str, default=None)
    parser.add_argument("--evidence_top_k", type=int, default=5, help="Number of documents to use for context.")

    # Dataset and save paths
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--ideal_number_tokens", type=int, default=2000)
    return parser.parse_known_args()

# Finetune 
def parse_finetuning_args():
    parser = argparse.ArgumentParser(description="Argument Parser for finetuning task")
    
    parser.add_argument("--save_data_path", type=str, required=True)    
    parser.add_argument("--save_model_path", type=str, required=True)
    
    # PEFT parameters
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--bias", type=str, default="none")
    parser.add_argument("--task_type", type=str, default="CAUSAL_LM")

    # SFTT parameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument("--evalution_strategy", type=str, default="steps")
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant")
    parser.add_argument("--disable-tqdm", action="store_true")

    return parser.parse_known_args()

# Evaluation
def parse_evaluation_args():
    parser = argparse.ArgumentParser(description="Argument Parser for evaluation task")

    # Metric, Hugging Face token or model path in case metric needs an LLM
    parser.add_argument("--metric", type=str, default="GLEU")
    parser.add_argument("--hf_token",  nargs='?', const='', required=False)
    parser.add_argument("--eval_model_path", type=str, default=None, required=False)

    # Save paths
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--save_every", type=int, default=10)
    return parser.parse_known_args()


def extract_responses(outputs, delimiter):
    return [x.split(delimiter)[1].strip() for x in outputs]

def get_derangement(sequence):
    valid_permutations = set([s for s in itertools.permutations(sequence) if not any([a == b for a, b in zip(s, sequence)])])

    return random.choice(list(valid_permutations))

def convert_to_hf_dataset(dataset):
    def gen():
        for item in dataset:
            yield {'qrd': item[0], 'e': item[1]}

    # Create a Dataset object from the generator function
    dset = Dataset.from_generator(gen)
    return dset

def load_latest_checkpoint(checkpoint_dir):
    checkpoint_folders = [folder for folder in os.listdir(checkpoint_dir)]
    sorted_checkpoint_folders = sorted(checkpoint_folders, key=lambda x: int(x.split("checkpoint")[1]))
    # Get the latest checkpoint folder
    latest_checkpoint_folder = sorted_checkpoint_folders[-1]
    return latest_checkpoint_folder

def plot_training_curve(data):
    all_losses = []
    for i in range (len(data)-1):
        all_losses.append(data[i]["loss"])
    epochs = list(range(len(data)-1))

    # Plot loss curve
    plt.plot(epochs, all_losses, marker='o')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_curve.png')
