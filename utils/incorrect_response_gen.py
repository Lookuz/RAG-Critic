import torch
import torch.nn.functional as F
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
):
    # Construct prompt
    inputs = [construct_prompt_for_incorrect_response(q, r) for q, r in inputs]
    tokenizer_inputs = [{'role': 'user', 'content': m} for m in inputs]

    # Tokenize inputs
    model_inputs = [tokenizer.apply_chat_template([chat], return_tensors="pt")
                    for chat in tokenizer_inputs]

    max_length = max([tensor.size(1) for tensor in model_inputs])

    # Pad tensors with zeros
    padded_tensors = []
    for tensor in model_inputs:
        padding = (0, max_length - tensor.size(1))
        padded_tensor = F.pad(tensor, padding, "constant", 0)
        padded_tensors.append(padded_tensor)

    # Stack padded tensors
    model_inputs = torch.stack(padded_tensors, dim=0).squeeze(1)

    generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    return utils.extract_mistral_responses(decoded)


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
