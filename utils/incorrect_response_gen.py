from transformers import GenerationConfig

from utils import const
from utils import utils


class TaskPrompt:
    """
    An object representation of a task-specific prompt
    """

    def __init__(self, instruction, template, delimiter, task) -> None:
        self.instruction = instruction
        self.template = template
        self.delimiter = delimiter
        self.task = task

    def construct(self, question, answer):
        return self.template.format(self.instruction, question, answer)


def get_prompt_from_task(task):

    task_prompts = {
        const.BOOTSTRAP_INCORRECT_RESPONSE_TASK: TaskPrompt(
            const.INCORRECT_RESPONSE_INSTRUCTION,
            const.INCORRECT_RESPONSE_TEMPLATE,
            const.INCORRECT_RESPONSE_DELIMITER,
            task
        )
    }

    return task_prompts[task]


def construct_prompt_for_incorrect_response(question, response):
    """
    Constructs the prompt to the model to generate a false response, under the provided question,
    as well as the correct response
    """
    return const.INCORRECT_RESPONSE_TEMPLATE.format(const.INCORRECT_RESPONSE_INSTRUCTION, question, response)


def generate_incorrect_responses(
    model, tokenizer,
    prompt: TaskPrompt, inputs,
    generation_config: GenerationConfig
):
    # Construct prompt
    inputs = [construct_prompt_for_incorrect_response(q, r) for q, r in inputs]

    # Tokenize inputs
    tokenized_inputs = tokenizer(inputs, padding="longest", add_special_tokens=True, return_tensors="pt")
    input_ids, attention_mask = tokenized_inputs["input_ids"].to(
        model.device), tokenized_inputs["attention_mask"].to(model.device)

    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return utils.extract_responses(outputs, prompt.delimiter)


# def generate_responses(
#     model, tokenizer,
#     prompt: TaskPrompt, inputs,
#     generation_config: GenerationConfig
# ):
#     # Construct prompt
#     inputs = [
#         prompt.construct(question, evidence, answer) for question, answer, evidence in inputs
#     ]

#     # Tokenize inputs
#     inputs = tokenizer(inputs, padding="longest",
#                        add_special_tokens=True, return_tensors="pt")
#     input_ids, attention_mask = inputs["input_ids"].to(
#         model.device), inputs["attention_mask"].to(model.device)

#     outputs = model.generate(
#         input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config)
#     outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     outputs = [(q, r, r_) for (q, r), r_ in zip(
#         inputs, utils.extract_responses(outputs, prompt.delimiter))]

#     return outputs
