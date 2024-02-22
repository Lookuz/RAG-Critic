# Tasks
BOOTSTRAP_INCORRECT_RESPONSE_TASK = "bootstrap-incorrect-response"
BOOTSTRAP_WRONG_CONTEXT_TASK = "bootstrap-wrong-context"
FINETUNE_CRITIC_TASK = "finetune-critic"

# Task-specific prompts
INCORRECT_RESPONSE_INSTRUCTION = "Provide an incorrect response to the following question, under the context provided. The correct response is supplied below."
INCORRECT_RESPONSE_TEMPLATE = "{}\n\n### QUESTION:\n{}\n\n### CONTEXT\n{}\n\n### CORRECT RESPONSE:\n{}\n\n### INCORRECT RESPONSE:\n"
INCORRECT_RESPONSE_DELIMITER = "### INCORRECT RESPONSE:\n"
