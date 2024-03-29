import os
import json
from typing import Union

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import pipeline, DPRReader, DPRReaderTokenizer
from langchain.schema.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

from utils.prompt import TaskPrompt, generate_responses, summarize_document, extract_snippet
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
    else:
        raise AssertionError("Provided task is not valid!")

def bootstrap_incorrect_responses(
    prompt : TaskPrompt, 
    ideal_number_tokens,
    dataset, data_path, dataset_args : dict,
    model, tokenizer,
    generation_config,
    batch_size,
    save_path,
    num_workers=1,
    save_every=1,
    mode="snippet",
    *args,
    **kwargs
):
    # Build dataset from original examples
    dataset = ContextualizedQADatasetForBootstrapping.from_dataset(dataset=dataset, data_path=data_path, **dataset_args)
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
        if i < num_generated:
            continue

        # Simplify evidence documents
        if mode == "summarize":
            batch = [(question, answer, '\n'.join(
                summarize_document(summarizer=summarize_chain, documents=evidence, tokenizer=tokenizer, ideal_number_tokens=ideal_number_tokens)
            )) for question, answer, evidence in batch]
        elif mode == "snippet":
            batch = [(question, answer, '\n'.join(
                extract_snippet(model_reader, tokenizer_reader, question, evidence)
            )) for question, answer, evidence in batch]

        # Generate responses using summarized evidence
        outputs = generate_responses(
            model, tokenizer, prompt, batch, generation_config
        )
        # Add additional responses to existing examples
        for q, r, d, r_ in outputs:
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
        if i < num_generated:
            continue

        # Check if evidence is already condensed, else perform some form of summarization
        if isinstance(batch[0][2], dict):
            batch = [(question, answer, '\n'.join(
                extract_snippet(model_reader, tokenizer_reader, question, evidence)
            ), generated) for question, answer, evidence, generated in batch]

        # Generate evaluation under correct context for correct responses
        inputs_correct_context = [(q, r, d) for q, r, d, _ in batch]
        outputs = generate_responses(
            model, tokenizer, prompt[EVALUATION_GENERATION_CORRECT_CASE], inputs_correct_context, generation_config
        )
        for (q, r, d), e in zip(inputs_correct_context, outputs):
            bootstrapped_examples.append({
                "question" : q, "answer" : r, "evidence" : d, "evaluation" : e
            })

        # Generate evaluation under correct context for wrong responses
        inputs_incorrect_response = [(q, r_, d) for q, _, d, r_ in batch]
        outputs = generate_responses(
            model, tokenizer, prompt[EVALUATION_GENERATION_WRONG_RESPONSE_CASE], inputs_incorrect_response, generation_config
        )
        for (q, r, d), e in zip(inputs_incorrect_response, outputs):
            bootstrapped_examples.append({
                "question" : q, "answer" : r, "evidence" : d, "evaluation" : e
            })
        
        # Generate evaluation under wrong context
        evidence_idx = get_derangement(list(range(len(batch))))
        # Randomly select evidence from other examples within batch
        inputs_incorrect_context = [(q, r, batch[i][2]) for i, (q, r, _, _) in zip(evidence_idx, batch)]
        outputs = generate_responses(
            model, tokenizer, prompt[EVALUATION_GENERATION_WRONG_CONTEXT_CASE], inputs_incorrect_context, generation_config
        )
        for (q, r, d), e in zip(inputs_incorrect_context, outputs):
            bootstrapped_examples.append({
                "question" : q, "answer" : r, "evidence" : d, "evaluation" : e
            })

        if (i + 1) % save_every == 0:
            # Save additional examples to new data files
            with open(save_path, "w") as f:
                json.dump(bootstrapped_examples, f, indent=4)

# Not-contextualized
class QADataset(Dataset):
    """
    Dataset for text in the form of [Q, R] pairs, containing the question and response respectively.
    Format of data: Each entry should be in the form {"question" : ..., "answer" : ...}
    """
    def __init__(
        self, data
    ) -> None:
        super().__init__()

        self.data = data

    def __getitem__(self, index):
        return (self.data[index]["question"], self.data[index]["answer"])
    
    def __len__(self):
        return len(self.data)
    
    @classmethod
    def from_dataset(cls, dataset, data_path, **kwargs):
        return {
            "triviaqa" : cls.from_trivia_qa
            # TODO: Other datasets, if needed
        }[dataset](data_path)
    
    @classmethod
    def from_trivia_qa(cls, data_path):
        with open(data_path, "r") as f:
            data = json.load(f)["Data"]

        examples = []
        for x in data:
            question, answer = x["Question"], x["Answer"]["Value"]
            examples.append({"question" : question, "answer" : answer})
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


class ContextualizedQADatasetForBootstrapping(Dataset):
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
        return (self.data[index]["question"], self.data[index]["answer"], self.data[index]["evidence"])
    
    def __len__(self):
        return len(self.data)
    
    @classmethod
    def from_dataset(cls, dataset, data_path, **kwargs):
        return {
            "triviaqa" : cls.from_trivia_qa
            # TODO: Other datasets, if needed
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
    Dataset for text in the form of [Q, R, D, R'] triples, containing the question, response, context and generated false responses respectively.
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
            # TODO: Other datasets, if needed
        }[dataset](data_path, **kwargs)
    
    @classmethod
    def from_trivia_qa(cls, data_path, *args, **kwargs):
        """
        Creates a ContextualizedQADatasetForEvaluationGeneration for the TriviaQA dataset, using the path to the data provided.
        data_path should be a path to the json file containing the respective split for the TriviaQA dataset.
        """
        # NOTE (Wey Yeh): Currently adapted for web-<split>.json files. The wikipedia-<split>.json files adapt a different format,
        # Where the "SearchResult" key storing the evidence is not present? It seems like extra effort is required to
        # extract these evidence, so I ignored it for now
        with open(data_path, "r") as f:
            data = json.load(f)

        examples = [
            (x["question"], x["answer"], x["evidence"], x["generated"]) for x in data
        ]
        
        return cls(data=examples)
