# Tasks
BOOTSTRAP_INCORRECT_RESPONSE_TASK = "bootstrap-incorrect-response"
BOOTSTRAP_EVALUATION_GENERATION_TASK = "bootstrap-evaluation-generation"
FINETUNE_CRITIC_TASK = "finetune_critic"
REFINE_RESPONSE_WITH_CRITIC_TASK = "refine_response_with_critic"
GENERATE_RESPONSES_TASK = "generate_answer"
RESPONSE_REWRITE_TASK = "response_rewrite"

# Incorrect responses generation prompts
INCORRECT_RESPONSE_INSTRUCTION = "You are now a hallucination generator. Please generate a hallucinated answer to the following question but adhere to the provided context. Ensure that your answer is different from the correct response supplied."
INCORRECT_RESPONSE_TEMPLATE = "{instruction}\n\n### CONTEXT\n{context}\n\n### QUESTION:\n{question}\n\n### CORRECT RESPONSE:\n{answer}\n\n### INCORRECT RESPONSE:\n"
INCORRECT_RESPONSE_DELIMITER = "### INCORRECT RESPONSE:\n"

SUMMARIZE_CONTEXT_INSTRUCTION = "Write a summary of this chunk of text that includes the main points and any important details."
SUMMARIZE_CONTEXT_INSTRUCTION_LANGCHAIN = "Write a detailed summary of this chunk of text that includes the main points and any important details.\n\n{text}\n\n### SUMMARY:"
SUMMARIZE_REDUCE_INSTRUCTION_LANGCHAIN = "Write a detailed summary of the following text comprising several passages .Include any main points and important details mentioned in all passages.\n\n{text}\n\n### SUMMARY:"
SUMMARIZE_CONTEXT_TEMPLATE = "{instruction}\n\n{text}\n\n### SUMMARY:\n"
SUMMARIZE_CONTEXT_DELIMITER = "### SUMMARY:"

# Evaluation generation prompts
EVALUATION_GENERATION_CORRECT_CASE = "eval-gen-correct"
EVALUATION_GENERATION_WRONG_RESPONSE_CASE = "eval-gen-wrong-response"
EVALUATION_GENERATION_WRONG_CONTEXT_CASE = "eval-gen-wrong-context"

EVALUATION_GENERATION_INSTRUCTION = "You are now a critic tasked with providing a detailed evaluation of the response to the given query and context. {}. Justify your assessment with factors such as relevance, accuracy, completeness and coherence."
EVALUATION_GENERATION_TEMPLATE = "{instruction}\n\n### CONTEXT\n{context}\n\n### QUESTION:\n{question}\n\n### RESPONSE:\n{answer}\n\n### EVALUATION:\n"
EVALUATION_GENERATION_DELIMITER = "### EVALUATION:"

EVALUATION_GENERATION_CORRECT_FILLER = "The response to the question provided below is correct. Elaborate on why this is so, keeping the context in mind"
EVALUATION_GENERATION_WRONG_CONTEXT_FILLER = "The context provided below is irrelevant to both the question. Elaborate why the context is unimportant for addressing the query. Explain why the response is wrong as a consequence of such irrelevant context"
EVALUATION_GENERATION_WRONG_RESPONSE_FILLER = "The response to the question provided below is incorrect. Elaborate on why this is so, keeping the context in mind"

# Critic fine-tuning/feedback prompts
CRITIC_FEEDBACK_INSTRUCTION = "You are now a critic tasked with providing a detailed evaluation of the response to the given query and context. Justify your assessment with factors such as relevance, accuracy, completeness and coherence."
CRITIC_FEEDBACK_TEMPLATE = EVALUATION_GENERATION_TEMPLATE
CRITIC_FEEDBACK_DELIMITER = EVALUATION_GENERATION_DELIMITER

# Standard generaton prompt
ANSWER_GENERATION_INSTRUCTION = "Please answer the following question, using the provided context as required."
ANSWER_GENERATION_TEMPLATE = "{instruction}\n\n### CONTEXT\n{context}\n\n### QUESTION:\n{question}\n\n### RESPONSE:"
ANSWER_GENERATION_DELIMITER = "### RESPONSE:"

# Response rewrite prompt using feedback critic
RESPONSE_REWRITE_INSTRUCTION = "Refine the response to the following question under the context provided, using the feedback given."
RESPONSE_REWRITE_TEMPLATE = "{instruction}\n\n### CONTEXT\n{context}\n\n### QUESTION:\n{question}\n\n### RESPONSE:{answer}\n\n### FEEDBACK:{evaluation}\n\n### REFINED RESPONSE:"
RESPONSE_REWRITE_DELIMITER = "### REFINED RESPONSE:"
