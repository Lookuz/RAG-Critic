from transformers import GenerationConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter


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
        ),
        # Wrong context task
        BOOTSTRAP_EVALUATION_GENERATION_TASK : {
            EVALUATION_GENERATION_CORRECT_CASE : TaskPrompt(
                EVALUATION_GENERATION_INSTRUCTION.format(EVALUATION_GENERATION_CORRECT_FILLER),
                EVALUATION_GENERATION_TEMPLATE,
                EVALUATION_GENERATION_DELIMITER
            ),
            EVALUATION_GENERATION_WRONG_RESPONSE_CASE : TaskPrompt(
                EVALUATION_GENERATION_INSTRUCTION.format(EVALUATION_GENERATION_WRONG_CONTEXT_FILLER),
                EVALUATION_GENERATION_TEMPLATE,
                EVALUATION_GENERATION_DELIMITER
            ),
            EVALUATION_GENERATION_WRONG_CONTEXT_CASE : TaskPrompt(
                EVALUATION_GENERATION_INSTRUCTION.format(EVALUATION_GENERATION_WRONG_RESPONSE_FILLER),
                EVALUATION_GENERATION_TEMPLATE,
                EVALUATION_GENERATION_DELIMITER
            ),
        }
        # TODO: Finetune critic task
    }

    return task_prompts[task]

def construct_prompt_for_incorrect_response(question, context, response):
    """
    Constructs the prompt to the model to generate a false response, under the provided question and context,
    as well as the correct response
    """
    return INCORRECT_RESPONSE_TEMPLATE.format(INCORRECT_RESPONSE_INSTRUCTION, question, context, response)

def construct_prompt_for_summarization(evidence=None):
    """
    Constructs the prompt for asking a model to summarize a piece of document or chunk
    """
    prompt = SUMMARIZE_CONTEXT_TEMPLATE.format(SUMMARIZE_CONTEXT_INSTRUCTION, evidence) if evidence is not None else SUMMARIZE_CONTEXT_TEMPLATE.format(SUMMARIZE_CONTEXT_INSTRUCTION)

    return prompt

def summarize_document(summarizer, documents):
    # Load each evidence document
    summaries = []
    for doc in documents:
        with open(doc, "r") as f:
            evidence = f.read()
    
        # Use LangChain text splitter to chunk the text
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=8000, chunk_overlap=500)
        evidence = text_splitter.create_documents([evidence])

        # Prompt model for summarization
        summary = summarizer.invoke(evidence)['output_text']
        summaries.append(summary)

    return summaries

def generate_responses(
        model, tokenizer, 
        prompt : TaskPrompt, inputs, 
        generation_config : GenerationConfig
    ):
    assert (inputs is not None and len(inputs) > 0)
    # (Q, R, D) case for bootstrapping or evaluation generation
    if len(inputs[0]) > 2:
        # Construct prompt
        inputs = [
            prompt.construct(question, evidence, answer) for question, answer, evidence in inputs
        ]

        # Tokenize inputs
        inputs = tokenizer(inputs, padding="longest", add_special_tokens=True, return_tensors="pt")
        input_ids, attention_mask = inputs["input_ids"].to(model.device), inputs["attention_mask"].to(model.device)

        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs = [(q, r, d, r_) for (q, r, d), r_ in zip(inputs, extract_responses(outputs, prompt.delimiter))]
    
    # (Q, D) case, for RAG output generation by the target generative model
    else:
        pass

    return outputs
