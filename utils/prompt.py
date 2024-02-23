from transformers import GenerationConfig

from utils.const import *
from utils.utils import extract_responses

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
    return INCORRECT_RESPONSE_TEMPLATE.format(INCORRECT_RESPONSE_INSTRUCTION, question, context, response)

def construct_prompt_for_summarization(evidence):
    """
    Constructs the prompt for asking a model to summarize a piece of document
    """
    return SUMMARIZE_CONTEXT_TEMPLATE.format(SUMMARIZE_CONTEXT_INSTRUCTION, evidence)

def generate_responses(
        model, tokenizer, 
        prompt : TaskPrompt, inputs : tuple, 
        generation_config : GenerationConfig
    ):
    # Summarize input documents
    summarized = []
    for q, r, d in inputs:
        # Load each documents
        evidence = []
        for doc in d:
            with open(doc, "r") as f:
                evidence.append(f.read())

        # Construct prompt for summarization
        evidence_prompts = [construct_prompt_for_summarization(e) for e in evidence]

        # Prompt model for summarization
        summaries = []
        for evidence in evidence_prompts:
            inputs = tokenizer(evidence, padding="longest", add_special_tokens=True, return_tensors="pt")
            input_ids, attention_mask = inputs["input_ids"].to(model.device), inputs["attention_mask"].to(model.device)
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config)
            summaries.append(tokenizer.batch_decode(outputs, skip_special_tokens=True))

        summarized.append(q, r, summaries)

    # Format input data using the prompt template
    inputs = [(question, answer, '\n'.join(evidence)) for question, answer, evidence in summarized]
    inputs = [
        prompt.construct(question, evidence, answer) for question, answer, evidence in inputs
    ]

    # Tokenize inputs
    inputs = tokenizer(inputs, padding="longest", add_special_tokens=True, return_tensors="pt")
    input_ids, attention_mask = inputs["input_ids"].to(model.device), inputs["attention_mask"].to(model.device)

    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    outputs = [(q, r, d, r_) for (q, r, d), r_ in zip(summarized, extract_responses(outputs, prompt.delimiter))]

    return outputs
