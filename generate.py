import os
import json

from tqdm import tqdm
from transformers import pipeline, DPRReader, DPRReaderTokenizer
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

from utils.const import *
from utils.prompt import *
from triviaqa_datasets.datasets import (
    ContextualizedQADatasetForGeneration,
    ContextualizedQADatasetForEvaluationGeneration,
    ContextualizedQADatasetForKeywordExtraction,
    ContextualizedQADataLoader,
)

def generate_answers(
    task, prompt : TaskPrompt, 
    dataset, data_path, dataset_args : dict,
    generation_config,
    batch_size,
    save_path,
    num_workers=1,
    save_every=1,
    model=None, tokenizer=None, 
    critic_model=None, critic_tokenizer=None,
    critic_prompt=None, rewrite_prompt=None,
    mode="snippet",
    ideal_number_tokens=2000
):
    if task == REFINE_RESPONSE_WITH_CRITIC_TASK:
        assert (critic_model is not None and critic_tokenizer is not None), "Critic model must be specified for generation with critic!"
        # Build dataset for generation
        dataset = ContextualizedQADatasetForEvaluationGeneration.from_dataset(dataset=dataset, data_path=data_path, **dataset_args)
        dataloader = ContextualizedQADataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    
    else: #GENERATE_RESPONSES_TASK
        # Build dataset for generation
        dataset = ContextualizedQADatasetForGeneration.from_dataset(dataset=dataset, data_path=data_path, **dataset_args)
        dataloader = ContextualizedQADataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    # Check for existing generated examples
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            bootstrapped_examples = json.load(f)
        num_generated = len(bootstrapped_examples)
    else:
        bootstrapped_examples, num_generated = [], 0

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

    generated = []
    for i, batch in enumerate(tqdm(dataloader)):
        if i < num_generated//batch_size:
            continue
        
        # Standard generation
        if task != REFINE_RESPONSE_WITH_CRITIC_TASK:
            # Simplify content of context documents to reduce context length
            if mode == "summarize":
                inputs = [(question, '\n'.join(
                    summarize_document(summarizer=summarize_chain, documents=evidence, tokenizer=tokenizer, ideal_number_tokens=ideal_number_tokens)
                )) for question, evidence, answer in batch]
            elif mode == "snippet":
                inputs = [(question, '\n'.join(
                    extract_snippet(model_reader, tokenizer_reader, question, evidence)
                )) for question, evidence, answer in batch]
            
            answers = [answer for _, _, answer in batch]

            # Generate initial responses using (Q, D)
            outputs = generate_responses(model, tokenizer, prompt, inputs, generation_config)
            generated.extend([{
                "question" : q, "answer" : a, "evidence" : d, "generated" : r
            } for (q, d, r), a in zip(outputs, answers)])
        
        elif task == REFINE_RESPONSE_WITH_CRITIC_TASK:
            # Generate evaluation for generated responses on test set
            inputs_generated_response = [(q, d, r_generated) for q, d, _, r_generated in batch]
            answers = [answer for _, _, answer, _ in batch]

            # Provide critic feedback for (Q, D, R) with critic model => (Q, D, R, E)
            outputs_critic = generate_responses(
                critic_model, critic_tokenizer, critic_prompt, inputs_generated_response, generation_config)
                
            # # Re-generate with critic model feedback - (Q, D, R, E)
            outputs = generate_responses(model, tokenizer, rewrite_prompt, outputs_critic, generation_config)

            generated.extend([{
                "question" : q, "answer" : a, "evidence" : d, "generated" : r, "refined" : r_, "evaluation" : e
            } for (q, d, r, e, r_), a in zip(outputs, answers)])

        else:
            raise AssertionError(f"Task {args.task} invalid!")


        if (i + 1) % save_every == 0:
            # Save additional examples to new data files
            with open(save_path, "w") as f:
                json.dump(generated, f, indent=4)

def extract_answers(
    task, prompt : TaskPrompt, 
    dataset, data_path, dataset_args : dict,
    generation_config,
    batch_size,
    save_path,
    num_workers=1,
    save_every=1,
    model=None, tokenizer=None, 
):
    # Build dataset for generation
    # [Q, R_zs, R_cr]
    dataset = ContextualizedQADatasetForKeywordExtraction.from_dataset(dataset=dataset, data_path=data_path, **dataset_args)
    dataloader = ContextualizedQADataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    # Check for existing generated examples
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            bootstrapped_examples = json.load(f)
        num_generated = len(bootstrapped_examples)
    else:
        bootstrapped_examples, num_generated = [], 0

    keywords = []
    for i, batch in enumerate(tqdm(dataloader)):
        if i < num_generated//batch_size:
            continue

        # Extract keywords from zero_shot responses
        inputs_keywords_zero_shot = [(q, r_zs) for q, r_zs, _ in batch]
        # outputs_zs: q,r_zs,r_zs_keywords
        outputs_zs = generate_responses(
            model, tokenizer, prompt, inputs_keywords_zero_shot, generation_config
        )

        # Extract keywords from critic-refined responses
        inputs_keywords_critic_refined = [(q, r_cr) for q, _, r_cr in batch]
        
        # outputs_cr: q,r_cr,keywords_cr
        outputs_cr = generate_responses(
            model, tokenizer, prompt, inputs_keywords_critic_refined, generation_config
        )

        keywords.extend([{
            "question" : q, "generated" : r_zs, "generated_keywords" : r_zs_keywords
        } for (q, r_zs, r_zs_keywords) in outputs_zs])

        keywords.extend([{
            "question" : q, "refined" : r_cr, "refined_keywords" : r_cr_keywords
        } for (q, r_cr, r_cr_keywords) in outputs_cr])        

        if (i + 1) % save_every == 0:
            # Save additional examples to new data files
            with open(save_path, "w") as f:
                json.dump(keywords, f, indent=4)
