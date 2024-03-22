from transformers import GenerationConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter


from utils.const import *
from utils.utils import extract_responses
from utils.latent_semantic_analysis import truncate_text

class TaskPrompt:
    """
    An object representation of a task-specific prompt
    """
    def __init__(self, instruction, template, delimiter) -> None:
        self.instruction = instruction
        self.template = template
        self.delimiter = delimiter

    def construct(self, question, context, answer):
        return self.template.format(self.instruction, question, context, answer)

def get_prompt_from_task(task):

    task_prompts = {
        BOOTSTRAP_INCORRECT_RESPONSE_TASK : TaskPrompt(
            INCORRECT_RESPONSE_INSTRUCTION,
            INCORRECT_RESPONSE_TEMPLATE,
            INCORRECT_RESPONSE_DELIMITER
        )
        # TODO: Wrong context task
        # TODO: Finetune critic task
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
    prompt = SUMMARIZE_CONTEXT_TEMPLATE.format(SUMMARIZE_CONTEXT_INSTRUCTION, evidence) if evidence is not None else SUMMARIZE_CONTEXT_TEMPLATE.format(SUMMARIZE_CONTEXT_INSTRUCTION)

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
        print(f"Summary: {summary}")
        summaries.append(summary)

    return summaries


def generate_responses(
        model, tokenizer, 
        prompt : TaskPrompt, inputs, 
        generation_config : GenerationConfig
    ):

    # inputs Q,A,E (E is converted into a string by ''.join(list))
    # Construct prompt
    inputs_prompt = [
        prompt.construct(question, evidence, answer) for question, answer, evidence in inputs
    ]

    # Tokenize inputs
    inputs_tokenized = tokenizer(inputs_prompt, padding="longest", add_special_tokens=True, return_tensors="pt")
    input_ids, attention_mask = inputs_tokenized["input_ids"].to(model.device), inputs_tokenized["attention_mask"].to(model.device)

    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    print(f"inputs: {inputs}")
    # inputs = [(question, answer, evidence) for question, answer, evidence in inputs]
    outputs = [(q, r, d, r_) for (q, r, d), r_ in zip(inputs, extract_responses(outputs, prompt.delimiter))]

    return outputs
