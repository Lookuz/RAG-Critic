import os
import json
from tqdm import tqdm
from nltk.translate.gleu_score import sentence_gleu
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from triviaqa_datasets.datasets import ContextualizedQADatasetForQualityEvaluation, ContextualizedQADataLoader


class DeepEvalLlama2_7B(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        tokenizer,
        device
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        model.to(self.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Llama 2 7B"


# Given a question, ground truth answer, evaluate the relevance of the zero-shot and critic-refined generated responses
def evaluate_answers_quality(
    metric,
    dataset, data_path,
    batch_size,
    save_path,
    auth_token,
    num_workers=1,
    save_every=1,
    device=None,
    *args, **kwargs
):

    dataset = ContextualizedQADatasetForQualityEvaluation.from_dataset(dataset=dataset, data_path=data_path)
    dataloader = ContextualizedQADataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    if metric == "GEval":
        if auth_token is not None:
            model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=auth_token)
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=auth_token)
        elif "evel_model_path" in kwargs:
            model = AutoModelForCausalLM.from_pretrained(kwargs["eval_model_path"])
            tokenizer = AutoTokenizer.from_pretrained(kwargs["eval_model_path"])
        else:
            raise AssertionError("GEval metric requires one of hf_token or eval_model_path")

        mistral_7b = DeepEvalLlama2_7B(model=model, tokenizer=tokenizer, device=device)

    # Check for existing generated examples
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            scores = json.load(f)
        num_generated = len(scores)
    else:
        scores, num_generated = [], 0

    for i, batch in enumerate(tqdm(dataloader)):
        if i < num_generated//batch_size:
            continue
        outputs = []
        # Generate evaluation under correct context for wrong responses
        if metric == 'GLEU':
            for (q, r_gt, r_zs, r_cr) in batch:
                # print('q', q)
                # print('gt:', r_gt)
                # print('r_zs', r_zs)
                # print('r_cr', r_cr)
                score_zs = sentence_gleu([r_gt.split()], r_zs.split())
                score_cr = sentence_gleu([r_gt.split()], r_cr.split())
                # print(score_zs, score_cr)
        elif metric == 'GEval':
            coherence_metric = GEval(
                name="Coherence",
                criteria="Coherence - determine if the actual output is matching with the expected output.",
                evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                model=mistral_7b,
            )
            for (q, r_gt, r_zs, r_cr) in batch:

                test_case_zs = LLMTestCase(
                    input=q,
                    actual_output=r_zs,
                    expected_output=r_gt
                )

                test_case_cr = LLMTestCase(
                    input=q,
                    actual_output=r_cr,
                    expected_output=r_gt
                )
            score_zs = coherence_metric.measure(test_case_zs)
            # print(coherence_metric.score)
            # print(coherence_metric.reason)

            score_cr = coherence_metric.measure(test_case_cr)
            # print(coherence_metric.score)
            # print(coherence_metric.reason)
        else:
            raise AssertionError(f"Metric {args.task} not implemented")

        outputs.append((q, r_gt, r_zs, r_cr, score_zs, score_cr))

        scores.extend([{
                "question": q,
                "answer": r_gt,
                "generated": r_zs,
                "score_generated": score_zs,
                "refined": r_cr,
                "score_refined": score_cr
            } for (q, r_gt, r_zs, r_cr,  score_zs, score_cr) in outputs])

        if (i + 1) % save_every == 0:
            # Save additional examples to new data files
            with open(save_path, "w") as f:
                json.dump(scores, f, indent=4)
