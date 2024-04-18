import os
import json
from typing import Union

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import pipeline, DPRReader, DPRReaderTokenizer
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

from utils.prompt import *
from utils.const import *
from utils.utils import get_derangement

def bootstrap_dataset(
    task,
    prompt : TaskPrompt, 
    dataset, data_path, dataset_args : dict,
    model, tokenizer,
    generation_config,
    batch_size,
    save_path,
    num_workers=1,
    save_every=1,
    *args,
    **kwargs
):
    if task == BOOTSTRAP_INCORRECT_RESPONSE_TASK:
        bootstrap_incorrect_responses(
            prompt, dataset=dataset, data_path=data_path, dataset_args=dataset_args,
            model=model, tokenizer=tokenizer, generation_config=generation_config, 
            batch_size=batch_size, save_path=save_path, num_workers=num_workers, 
            save_every=save_every, *args, **kwargs
        )
    elif task == BOOTSTRAP_EVALUATION_GENERATION_TASK:
        bootstrap_evaluation_generation(
            prompt, dataset=dataset, data_path=data_path, dataset_args=dataset_args,
            model=model, tokenizer=tokenizer, generation_config=generation_config, 
            batch_size=batch_size, save_path=save_path, num_workers=num_workers, 
            save_every=save_every, *args, **kwargs
        )
    elif task == GENERATE_RESPONSES_TASK:
        bootstrap_incorrect_responses(
            prompt, dataset=dataset, data_path=data_path, dataset_args=dataset_args,
            model=model, tokenizer=tokenizer, generation_config=generation_config, 
            batch_size=batch_size, save_path=save_path, num_workers=num_workers, 
            save_every=save_every, *args, **kwargs
        )
    else:
        raise AssertionError("Provided task is not valid!")

def bootstrap_incorrect_responses(
    prompt : TaskPrompt, 
    dataset, data_path, dataset_args : dict,
    model, tokenizer,
    generation_config,
    batch_size,
    save_path,
    num_workers=1,
    save_every=1,
    mode="snippet",
    ideal_number_tokens=2000,
    *args,
    **kwargs
):
    # Build dataset from original examples
    dataset = ContextualizedQADatasetForGeneration.from_dataset(dataset=dataset, data_path=data_path, **dataset_args)
    dataloader = ContextualizedQADataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    # Check for existing generated examples
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            bootstrapped_examples = json.load(f)
        num_generated = len(bootstrapped_examples)
    else:
        bootstrapped_examples, num_generated = [], 0

    assert mode in ["summarize", "snippet"], "mode argument must be one of 'summary' or 'snippet'!"

    if mode == "summarize":
        # Initialize summarizer from LangChain
        llm = HuggingFacePipeline(pipeline=pipeline(
            "text-generation",
            model=model, tokenizer=tokenizer,
            max_new_tokens=2048,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            repetition_penalty=generation_config.repetition_penalty,
            do_sample=generation_config.do_sample
        ))
        summarize_chain = load_summarize_chain(
            llm=llm, chain_type="map_reduce",
            map_prompt=PromptTemplate(template=SUMMARIZE_CONTEXT_INSTRUCTION_LANGCHAIN, input_variables=["text"]), # Prompt for summarizing a single chunk,
            combine_prompt=PromptTemplate(template=SUMMARIZE_REDUCE_INSTRUCTION_LANGCHAIN, input_variables=["text"]), # Prompt for reducing all summaries to a single summary
            token_max=1024
        )
    elif mode == "snippet":
        tokenizer_reader = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-multiset-base")
        model_reader = DPRReader.from_pretrained("facebook/dpr-reader-multiset-base").to(model.device)

    # Generate additional examples
    for i, batch in enumerate(tqdm(dataloader)):
        if i < num_generated//batch_size:
            continue

        # Simplify evidence documents
        if mode == "summarize":
            batch = [(question, '\n'.join(
                summarize_document(summarizer=summarize_chain, documents=evidence, tokenizer=tokenizer, ideal_number_tokens=ideal_number_tokens)
            ), answer) for question, evidence, answer in batch]
        elif mode == "snippet":
            batch = [(question, '\n'.join(
                extract_snippet(model_reader, tokenizer_reader, question, evidence)
            ), answer) for question, evidence, answer in batch]

        # Generate responses using summarized evidence
        outputs = generate_responses(
            model, tokenizer, prompt, batch, generation_config
        )

        # Add additional responses to existing examples
        for q, d, r, r_ in outputs:
            bootstrapped_examples.append({
                "question" : q, "answer" : r, "evidence" : d, "generated" : r_
            })

        if (i + 1) % save_every == 0:
            # Save additional examples to new data files
            with open(save_path, "w") as f:
                json.dump(bootstrapped_examples, f, indent=4)

    return bootstrapped_examples

def bootstrap_evaluation_generation(
    prompt : dict[TaskPrompt], 
    dataset, data_path, dataset_args : dict,
    model, tokenizer,
    generation_config,
    batch_size,
    save_path,
    num_workers=1,
    save_every=1,
    *args,
    **kwargs
):
    # Build dataset from original examples
    dataset = ContextualizedQADatasetForEvaluationGeneration.from_dataset(dataset=dataset, data_path=data_path, **dataset_args)
    dataloader = ContextualizedQADataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    tokenizer_reader = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-multiset-base")
    model_reader = DPRReader.from_pretrained("facebook/dpr-reader-multiset-base").to(model.device)

    # Check for existing generated examples
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            bootstrapped_examples = json.load(f)
        num_generated = len(bootstrapped_examples)
    else:
        bootstrapped_examples, num_generated = [], 0

    # Generate additional examples
    bootstrapped_examples = []
    for i, batch in enumerate(tqdm(dataloader)):
        if i < num_generated//batch_size:
            continue

        # Generate evaluation under correct context for correct responses
        inputs_correct_context = [(q, d, r) for q, d, r, _ in batch]
        outputs = generate_responses(
            model, tokenizer, prompt[EVALUATION_GENERATION_CORRECT_CASE], inputs_correct_context, generation_config
        )
        bootstrapped_examples.extend([{
            "question" : q, "answer" : r, "evidence" : d, "evaluation" : e
        } for (q, d, r, e) in outputs])

        # Generate evaluation under correct context for wrong responses
        inputs_incorrect_response = [(q, d, r_) for q, d, _, r_ in batch]
        outputs = generate_responses(
            model, tokenizer, prompt[EVALUATION_GENERATION_WRONG_RESPONSE_CASE], inputs_incorrect_response, generation_config
        )
        bootstrapped_examples.extend([{
            "question" : q, "answer" : r, "evidence" : d, "evaluation" : e
        } for (q, d, r, e) in outputs])
        
        if (i + 1) % save_every == 0:
            # Save additional examples to new data files
            with open(save_path, "w") as f:
                json.dump(bootstrapped_examples, f, indent=4)

# Fine-tuning dataset
class ContextualizedQADatasetForCriticFinetuning(Dataset):
    """
    Dataset for text in the form of [Q, R, D, E] pairs, containing the question and response respectively.
    Format of data: Each entry should be in the form {"question" : ..., "answer" : ..., "evidence" : ..., "evaluation" : ...}
    """
    def __init__(
        self, data, prompt : TaskPrompt = None
    ) -> None:
        super().__init__()
        if prompt is None:
            prompt = get_prompt_from_task(FINETUNE_CRITIC_TASK)
        self.prompt = prompt

        self.data = data

    def __getitem__(self, index):
        question, evidence, answer, evaluation = self.data[index]
        prompt = self.prompt.construct(question, evidence, answer)
        return (prompt, evaluation)
    
    def __len__(self):
        return len(self.data)
    
    @classmethod
    def from_dataset(cls, dataset, data_path, **kwargs):
        return {
            "triviaqa" : cls.from_trivia_qa
        }[dataset](data_path)
    
    @classmethod
    def from_trivia_qa(cls, data_path):
        with open(data_path, "r") as f:
            data = json.load(f)

        # NOTE: Following the same example format as per bootstrap_evaluation_generation saving, since we are fine-tuning
        # from examples generated using that function
        examples = [(x["question"], x["evidence"],  x["answer"], x["evaluation"]) for x in data]
        
        return cls(data=examples)

# Bootstrapping datasets
class ContextualizedQADatasetForGeneration(Dataset):
    """
    Dataset for text in the form of [Q, R, D] triples, containing the question, response and context respectively.
    Format of data: Each entry should be in the form {"question" : ..., "answer" : ..., "evidence : ...}
    """
    def __init__(
        self, data
    ) -> None:
        super().__init__()

        self.data = data

    def __getitem__(self, index):
        return (self.data[index]["question"], self.data[index]["evidence"], self.data[index]["answer"],)
    
    def __len__(self):
        return len(self.data)
    
    @classmethod
    def from_dataset(cls, dataset, data_path, **kwargs):
        return {
            "triviaqa" : cls.from_trivia_qa
        }[dataset](data_path, **kwargs)
    
    @classmethod
    def from_trivia_qa(cls, data_path, evidence_path=None, top_k=1):
        """
        Creates a ContextualizedQADatasetForBootstrapping for the TriviaQA dataset, using the path to the data provided.
        data_path should be a path to the json file containing the respective split for the TriviaQA dataset.
        """
        # NOTE (Wey Yeh): Currently adapted for web-<split>.json files. The wikipedia-<split>.json files adapt a different format,
        # Where the "SearchResult" key storing the evidence is not present? It seems like extra effort is required to
        # extract these evidence, so I ignored it for now
        with open(data_path, "r") as f:
            data = json.load(f)["Data"]

        examples = []
        for x in data:
            question, answer = x["Question"], x["Answer"]["Value"]
            if evidence_path is not None:
                # Retrieve necessary evidence documents
                evidence = [os.path.join(evidence_path, e['Filename']) for e in x["EntityPages"]]
                evidence = evidence[:top_k] if top_k is not None else evidence
            else:
                evidence = None

            # Store (Q, R, D) triple
            examples.append({"question" : question, "answer" : answer, "evidence" : evidence})

        return cls(data=examples)

class ContextualizedQADatasetForEvaluationGeneration(Dataset):
    """
    Dataset for text in the form of [Q, R, D, R'] triples, containing the question, response, context and generated (false) responses respectively.
    Format of data: Each entry should be in the form {"question" : ..., "answer" : ..., "evidence : ..., "generated" : ...}
    """
    def __init__(
        self, data
    ) -> None:
        super().__init__()

        self.data = data  

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    @classmethod
    def from_dataset(cls, dataset, data_path, **kwargs):
        return {
            "triviaqa" : cls.from_trivia_qa
        }[dataset](data_path, **kwargs)
    
    @classmethod
    def from_trivia_qa(cls, data_path, *args, **kwargs):
        """
        Creates a ContextualizedQADatasetForEvaluationGeneration for the TriviaQA dataset, using the path to the data provided.
        data_path should be a path to the json file containing the respective split for the TriviaQA dataset.
        """
        with open(data_path, "r") as f:
            data = json.load(f)

        examples = [
            (x["question"], x["evidence"],  x["answer"], x["generated"]) for x in data
        ]
        
        return cls(data=examples)

# Answers evaluation dataset: zero-shot vs critic-refined
class ContextualizedQADatasetForQualityEvaluation(Dataset):
    """
    Dataset for text in the form of [Q, R, R_zs, R_cr] triples, containing the question, the ground-truth answer,
    the zero-shot response, and the critic-refined response respectively.
    Format of data: Each entry should be in the form {"question" : ..., "answer" : ..., "generated" : ..., "refined" : ...}
    """
    def __init__(
        self, data
    ) -> None:
        super().__init__()

        self.data = data  

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    @classmethod
    def from_dataset(cls, dataset, data_path, **kwargs):
        return {
            "triviaqa" : cls.from_trivia_qa
        }[dataset](data_path, **kwargs)
    
    @classmethod
    def from_trivia_qa(cls, data_path, *args, **kwargs):
        """
        Creates a ContextualizedQADatasetForEvaluationGeneration for the TriviaQA dataset, using the path to the data provided.
        data_path should be a path to the json file containing the respective split for the TriviaQA dataset.
        """
        with open(data_path, "r") as f:
            data = json.load(f)

        examples = [
            (x["question"], x["answer"],  x["generated"], x["refined"]) for x in data
        ]
        
        return cls(data=examples)

class ContextualizedQADatasetForKeywordExtraction(Dataset):
    """
    Dataset for text in the form of [Q, R_zs, R_cr] , containing the question, the zero-shot response and the critic-refined response.
    Format of data: Each entry should be in the form {"question" : ..., "generated" : ..., "refined" : ...}
    """
    def __init__(
        self, data
    ) -> None:
        super().__init__()

        self.data = data  

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    @classmethod
    def from_dataset(cls, dataset, data_path, **kwargs):
        return {
            "triviaqa" : cls.from_trivia_qa
        }[dataset](data_path, **kwargs)
    
    @classmethod
    def from_trivia_qa(cls, data_path, *args, **kwargs):
        """
        Creates a ContextualizedQADatasetForKeywordExtraction for the TriviaQA dataset, using the path to the data provided.
        """
        with open(data_path, "r") as f:
            data = json.load(f)

        examples = [
            (x["question"], x["generated"], x["refined"]) for x in data
        ]
        
        return cls(data=examples)


class ContextualizedQADataLoader(DataLoader):
    def __init__(self, 
        dataset: Dataset, 
        batch_size: int = 1, 
        shuffle: bool = False, 
        num_workers: int = 0):
        
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=ContextualizedQADataLoader.collate_fn)
    
    @classmethod
    def collate_fn(cls, batch):
        return batch