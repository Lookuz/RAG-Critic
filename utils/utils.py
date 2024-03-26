import argparse
import itertools
import random

def parse_args():
    parser = argparse.ArgumentParser()
    # Model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        default="lmsys/vicuna-13b-v1.5-16k",
        # default="/home/users/nus/e1101650/scratch/vicuna-13b-16k-cache/models--lmsys--vicuna-13b-v1.5-16k/snapshots/17c61f9ca19f5a7a04e96b2cc0d9bcf2920cb8c2",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "xpu", "npu"],
        default="cuda",
        help="The device type",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--load-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--cpu-offloading",
        action="store_true",
        help="Only when using 8-bit quantization: Offload excess weights to the CPU that don't fit on the GPU",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-gpu-memory", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    # Generation arguments
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--do-sample", action='store_true')

    # Task-specific arguments
    parser.add_argument(
        "--task",
        type=str,
        choices=["bootstrap-incorrect-response", "bootstrap-wrong-context", "finetune-critic"],
    
    )
    parser.add_argument("--evidence-path", type=str, default=None)
    parser.add_argument("--evidence-top-k", type=int, default=5, help="Number of documents to use for context.")

    # Dataset and save paths
    parser.add_argument(
        "--dataset",
        type=str,
        default="triviaqa",
    )
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--ideal-number-tokens", type=int, required=True)

    return parser.parse_args()

def extract_responses(outputs, delimiter):
    return [x.split(delimiter)[1].strip() for x in outputs]

def get_derangement(sequence):
    valid_permutations = set([s for s in itertools.permutations(sequence) if not any([a == b for a, b in zip(s, sequence)])])

    return random.choice(list(valid_permutations))
