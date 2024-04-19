import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from nltk.translate.gleu_score import sentence_gleu
from transformers import AutoModelForCausalLM, AutoTokenizer
# from deepeval.models.base_model import DeepEvalBaseLLM
# from deepeval.metrics import GEval
# from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from triviaqa_datasets.datasets import ContextualizedQADatasetForQualityEvaluation, ContextualizedQADataLoader


# class DeepEvalLlama2_7B(DeepEvalBaseLLM):
#     def __init__(
#         self,
#         model,
#         tokenizer,
#         device
#     ):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.device = device

#     def load_model(self):
#         return self.model

#     def generate(self, prompt: str) -> str:
#         model = self.load_model()

#         model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
#         model.to(self.device)

#         generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
#         return self.tokenizer.batch_decode(generated_ids)[0]

#     async def a_generate(self, prompt: str) -> str:
#         return self.generate(prompt)

#     def get_model_name(self):
#         return "Llama 2 7B"


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
    print("Calculating similarity scores...")
    dataset = ContextualizedQADatasetForQualityEvaluation.from_dataset(dataset=dataset, data_path=data_path)
    dataloader = ContextualizedQADataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    if metric == 'GLEU':
        model = None
        prepare_input = prepare_GLEU_input

        def score(input_tuple):
            return sentence_gleu(input_tuple[0], input_tuple[1])

    elif metric == 'SentenceSimilarity':
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        prepare_input = prepare_SentenceSimilarity_input

        def score(input_tuple):
            return util.pytorch_cos_sim(input_tuple[0], input_tuple[1]).item()

    # elif metric == "GEval":
    #     if auth_token is not None:
    #         model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=auth_token)
    #         tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=auth_token)
    #     elif "evel_model_path" in kwargs:
    #         model = AutoModelForCausalLM.from_pretrained(kwargs["eval_model_path"])
    #         tokenizer = AutoTokenizer.from_pretrained(kwargs["eval_model_path"])
    #     else:
    #         raise AssertionError("GEval metric requires one of hf_token or eval_model_path")

    #     mistral_7b = DeepEvalLlama2_7B(model=model, tokenizer=tokenizer, device=device)
    #     coherence_metric = GEval(
    #             name="Coherence",
    #             criteria="Coherence - determine if the actual output is matching with the expected output.",
    #             evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    #             model=mistral_7b,
    #         )
    #     model = None
    #     prepare_input = prepare_GEval_input
    #     score = coherence_metric.measure

    else:
        raise AssertionError(f"Metric {metric} not implemented")

    scores = []

    for i, batch in enumerate(tqdm(dataloader)):
        outputs = []
        for (q, r_gt, r_zs, r_cr) in batch:
            input_zs, input_cr = prepare_input(r_gt=r_gt, r_zs=r_zs, r_cr=r_cr, q=q, model=model)
            score_zs = score(input_zs)
            score_cr = score(input_cr)

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


def prepare_GLEU_input(r_gt, r_zs, r_cr, q=None, model=None):
    r_gt = r_gt.lower()
    return ([r_gt.split()], r_zs.lower().split()), ([r_gt.split()], r_cr.lower().split())


# def prepare_GEval_input(r_gt, r_zs, r_cr, q=None, model=None):
#     test_case_zs = LLMTestCase(
#                     input=q,
#                     actual_output=r_zs,
#                     expected_output=r_gt
#                 )
#     test_case_cr = LLMTestCase(
#                     input=q,
#                     actual_output=r_cr,
#                     expected_output=r_gt
#                 )
#     return test_case_zs, test_case_cr


def prepare_SentenceSimilarity_input(r_gt, r_zs, r_cr, q=None, model=None):
    embedding_gt = model.encode(r_gt, convert_to_tensor=True)
    embedding_zs = model.encode(r_zs, convert_to_tensor=True)
    embedding_cr = model.encode(r_cr, convert_to_tensor=True)
    return (embedding_gt, embedding_zs), (embedding_gt, embedding_cr)
