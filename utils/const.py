# Tasks
BOOTSTRAP_INCORRECT_RESPONSE_TASK = "bootstrap-incorrect-response"
BOOTSTRAP_WRONG_CONTEXT_TASK = "bootstrap-wrong-context"
FINETUNE_CRITIC_TASK = "finetune-critic"

# Task-specific prompts
INCORRECT_RESPONSE_INSTRUCTION = "You are now a hallucination generator. Please generate a hallucinated answer to the following question but adhere to the provided context. \
    Ensure that your answer is different from the correct response supplied."
INCORRECT_RESPONSE_TEMPLATE = "{}\n\n### CONTEXT\n{}\n\n### QUESTION:\n{}\n\n### CORRECT RESPONSE:\n{}\n\n### INCORRECT RESPONSE:\n"
INCORRECT_RESPONSE_DELIMITER = "### INCORRECT RESPONSE:\n"

SUMMARIZE_CONTEXT_INSTRUCTION = "Write a summary of this chunk of text that includes the main points and any important details."
SUMMARIZE_CONTEXT_INSTRUCTION_LANGCHAIN = "Write a detailed summary of this chunk of text that includes the main points and any important details.\n\n{text}\n\n### SUMMARY:"
SUMMARIZE_REDUCE_INSTRUCTION_LANGCHAIN = "Write a detailed summary of the following text comprising several passages .Include any main points and important details mentioned in all passages.\n\n{text}\n\n### SUMMARY:"
SUMMARIZE_CONTEXT_TEMPLATE = "{}\n\n{text}\n\n### SUMMARY:\n"
SUMMARIZE_CONTEXT_DELIMITER = "### SUMMARY:"
