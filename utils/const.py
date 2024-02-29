# Tasks
BOOTSTRAP_INCORRECT_RESPONSE_TASK = "bootstrap-incorrect-response"
BOOTSTRAP_WRONG_CONTEXT_TASK = "bootstrap-wrong-context"
FINETUNE_CRITIC_TASK = "finetune-critic"

# Task-specific prompts
INCORRECT_RESPONSE_INSTRUCTION = "You are now a hallucination generator. Please generate a hallucinated answer to the following question but adhere to the provided context. \
    Ensure that your answer is different from the correct response supplied."
INCORRECT_RESPONSE_TEMPLATE = "{}\n\n### CONTEXT\n{}\n\n### QUESTION:\n{}\n\n### CORRECT RESPONSE:\n{}\n\n### INCORRECT RESPONSE:\n"
INCORRECT_RESPONSE_DELIMITER = "### INCORRECT RESPONSE:\n"

SUMMARIZE_CONTEXT_INSTRUCTION = "Summarize the following document concisely in under four sentences."
SUMMARIZE_CONTEXT_TEMPLATE = "{}\n{}\n\n### SUMMARY:\n"
SUMMARIZE_CONTEXT_DELIMITER = "### SUMMARY:"
