import torch
from transformers import GenerationConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.const import *
from utils.utils import extract_responses
from utils.latent_semantic_analysis import truncate_text

# POST_PROCESS_RESPONSE_INSTRUCTION = "Given the QUESTION, extract the keyword(s) in IMPROVED RESPONSE needed to answer the QUESTION."
# RESPONSE_REWRITE_TEMPLATE = "{instruction}\n\n### QUESTION:\n{question}\n\n### IMPROVED RESPONSE:{improved_answer}\n\n### KEYWORDS:"
# RESPONSE_REWRITE_DELIMITER = "### KEYWORDS:"

class TaskPrompt:
    """
    An object representation of a task-specific prompt
    """
    def __init__(self, instruction, template, delimiter) -> None:
        self.instruction = instruction
        self.template = template
        self.delimiter = delimiter

    def construct(self, question, context, answer=None, evaluation=None):
        if answer is not None:
            if evaluation is not None:
                return self.template.format(
                    instruction=self.instruction, question=question, context=context, answer=answer, evaluation=evaluation)
            else: # Evaluation generation prompts
                return self.template.format(instruction=self.instruction, question=question, context=context, answer=answer)
        else: # Standard generaton prompt
            return self.template.format(instruction=self.instruction, question=question, context=context)
    
    def construct_for_keyword_extraction(self, question, answer):
        return self.template.format(instruction=self.instruction, question=question, answer=answer)

def get_prompt_from_task(task):

    task_prompts = {
        # Incorrect response bootstrap task
        BOOTSTRAP_INCORRECT_RESPONSE_TASK : TaskPrompt(
            INCORRECT_RESPONSE_INSTRUCTION,
            INCORRECT_RESPONSE_TEMPLATE,
            INCORRECT_RESPONSE_DELIMITER
        ),
        # Evaluation generation task
        BOOTSTRAP_EVALUATION_GENERATION_TASK : {
            EVALUATION_GENERATION_CORRECT_CASE : TaskPrompt(
                EVALUATION_GENERATION_INSTRUCTION.format(filler=EVALUATION_GENERATION_CORRECT_FILLER),
                EVALUATION_GENERATION_TEMPLATE,
                EVALUATION_GENERATION_DELIMITER
            ),
            EVALUATION_GENERATION_WRONG_RESPONSE_CASE : TaskPrompt(
                EVALUATION_GENERATION_INSTRUCTION.format(filler=EVALUATION_GENERATION_WRONG_RESPONSE_FILLER),
                EVALUATION_GENERATION_TEMPLATE,
                EVALUATION_GENERATION_DELIMITER
            ),
        },
        # Standard generation
        GENERATE_RESPONSES_TASK : TaskPrompt(
            ANSWER_GENERATION_INSTRUCTION, 
            ANSWER_GENERATION_TEMPLATE, 
            ANSWER_GENERATION_DELIMITER
        ),
        # Critic feedback prompts
        REFINE_RESPONSE_WITH_CRITIC_TASK : TaskPrompt(
            CRITIC_FEEDBACK_INSTRUCTION,
            CRITIC_FEEDBACK_TEMPLATE,
            CRITIC_FEEDBACK_DELIMITER
        ),
        # Response refinement 
        RESPONSE_REWRITE_TASK : TaskPrompt(
            RESPONSE_REWRITE_INSTRUCTION,
            RESPONSE_REWRITE_TEMPLATE,
            RESPONSE_REWRITE_DELIMITER
        ),
        # Keyword extraction from refined response 
        POST_PROCESS_RESPONSE_TASK : TaskPrompt(
            POST_PROCESS_RESPONSE_INSTRUCTION,
            POST_PROCESS_RESPONSE_TEMPLATE,
            POST_PROCESS_RESPONSE_DELIMITER
        )
    }

    return task_prompts[task]

def construct_prompt_for_incorrect_response(question, context, response):
    """
    Constructs the prompt to the model to generate a false response, under the provided question and context,
    as well as the correct response
    """
    return INCORRECT_RESPONSE_TEMPLATE.format(INCORRECT_RESPONSE_INSTRUCTION, context, question, response)

def construct_prompt_for_summarization(evidence=None):
    """
    Constructs the prompt for asking a model to summarize a piece of document or chunk
    """
    prompt = SUMMARIZE_CONTEXT_TEMPLATE.format(
        instruction=SUMMARIZE_CONTEXT_INSTRUCTION, text=evidence
    ) if evidence is not None else SUMMARIZE_CONTEXT_TEMPLATE.format(SUMMARIZE_CONTEXT_INSTRUCTION)

    return prompt

def summarize_document(summarizer, documents, tokenizer, ideal_number_tokens):
    # Load each evidence document
    summaries = []
    for doc in documents:
        with open(doc, "r") as f:
            evidence = f.read()
    
        shortened_evidence = truncate_text(text=evidence, llm_max_tokens=ideal_number_tokens, hf_tokenizer=tokenizer)
        # Use LangChain text splitter to chunk the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=100)
        chunked_shortened_evidence = text_splitter.create_documents([shortened_evidence])

        # Prompt model for summarization
        summary = summarizer.invoke(chunked_shortened_evidence)['output_text']
        summaries.append(summary)

    return summaries

def extract_snippet(reader, tokenizer, question, documents):
    """
    Takes a set of evidence documents, and returns a set of snippets of the same size as documents
    containing short paragraph from each documents with the most relevant information according to the question.

    """
    snippets = []
    for document in documents:
        # Read context passage
        with open(document, "r") as f:
            evidence = f.read()

    
        # Use LangChain text splitter to chunk the text
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=2000, chunk_overlap=250)
        evidence = text_splitter.create_documents([evidence])
        evidence = [e.page_content for e in evidence]

        # Output the relevance score for each paragraph in document
        relevance_logits = []
        for e in evidence:
            encoded_inputs = tokenizer(
                questions=[question],
                texts=[e],
                truncation=True,
                return_tensors="pt",
            )
            encoded_inputs = encoded_inputs.to(reader.device)
            outputs = reader(**encoded_inputs)
            outputs.relevance_logits
            relevance_logits.append(outputs.relevance_logits)

        # Accept only the most relevant snippet
        relevance_logits = torch.cat(relevance_logits).flatten()
        top_idx = torch.argmax(relevance_logits)
        snippets.append(evidence[top_idx])

    return snippets

def generate_responses(
        model, tokenizer, 
        prompt : TaskPrompt, inputs, 
        generation_config : GenerationConfig
    ):

    if prompt.delimiter == "### FORMATTED RESPONSE:":
        # Construct prompt from inputs (Q, R)
        inputs_prompt = [prompt.construct_for_keyword_extraction(*x) for x in inputs]

    else:
        # Construct prompt from inputs (Q, D, [R, E])
        inputs_prompt = [prompt.construct(*x) for x in inputs]

    # Tokenize inputs
    tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs_tokenized = tokenizer(inputs_prompt, padding="longest", add_special_tokens=True, return_tensors="pt")
    input_ids, attention_mask = inputs_tokenized["input_ids"].to(model.device), inputs_tokenized["attention_mask"].to(model.device)

    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Combine outputs with inputs
    outputs = [(*x, y) for x, y in zip(inputs, extract_responses(outputs, prompt.delimiter))]
    
    return outputs
