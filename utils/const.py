# Tasks
BOOTSTRAP_INCORRECT_RESPONSE_TASK = "bootstrap-incorrect-response"
BOOTSTRAP_EVALUATION_GENERATION_TASK = "bootstrap-evaluation-generation"
FINETUNE_CRITIC_TASK = "finetune_critic"
REFINE_RESPONSE_WITH_CRITIC_TASK = "refine_response_with_critic"
GENERATE_RESPONSES_TASK = "generate_answer"
RESPONSE_REWRITE_TASK = "response_rewrite"
EVALUATE_ANSWERS_QUALITY_TASK = "evaluate_answers_quality"
POST_PROCESS_RESPONSE_TASK = "post-process"

# Incorrect responses generation prompts
INCORRECT_RESPONSE_INSTRUCTION = "Given the QUESTION, CONTEXT and CORRECT RESPONSE, write an INCORRECT RESPONSE that is different from the CORRECT RESPONSE. The INCORRECT RESPONSE must be false but related to the provided CONTEXT and is a possible answer to the QUESTION. Make sure you only return the INCORRECT RESPONSE and nothing more."
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

# EVALUATION_GENERATION_INSTRUCTION = "You are now a critic tasked with providing a detailed evaluation of the response to the given question and context. {}. Justify your assessment with factors such as relevance, accuracy, completeness and coherence."
EVALUATION_GENERATION_INSTRUCTION = "{filler}."
EVALUATION_GENERATION_TEMPLATE = "{instruction}\n\n### CONTEXT\n{context}\n\n### QUESTION:\n{question}\n\n### RESPONSE:\n{answer}\n\n### EVALUATION:\n"
EVALUATION_GENERATION_DELIMITER = "### EVALUATION:"

EVALUATION_GENERATION_CORRECT_FILLER = "The RESPONSE to the QUESTION is correct. Write one to two sentences to explain why the RESPONSE is correct based on the most relevant sentence in the given CONTEXT. Make sure you only return the explanation for the CORRECT RESPONSE and nothing more."
EVALUATION_GENERATION_WRONG_RESPONSE_FILLER = "The RESPONSE to the QUESTION is incorrect. Write one to two sentences to explain why the RESPONSE is incorrect based on the most relevant sentence in the given CONTEXT.  Support your explanation with one sentence on relevance and accuracy. Keep your explanation concise and to the point."

# Critic fine-tuning/feedback prompts
CRITIC_FEEDBACK_INSTRUCTION = "The RESPONSE to the QUESTION is <correct/incorrect>. Write one to two sentences to explain why the RESPONSE is <correct/incorrect> based on the most relevant sentence in the given CONTEXT.  Support your explanation with one sentence on relevance and accuracy. Keep your explanation concise and to the point."
CRITIC_FEEDBACK_TEMPLATE = EVALUATION_GENERATION_TEMPLATE
CRITIC_FEEDBACK_DELIMITER = EVALUATION_GENERATION_DELIMITER

# Standard generaton prompt
# ANSWER_GENERATION_INSTRUCTION = "Please answer the following question, using the provided context as required."
ANSWER_GENERATION_INSTRUCTION = "Given the QUESTION and CONTEXT, write a CORRECT RESPONSE found in the provided CONTEXT. Make sure you only return the CORRECT RESPONSE and nothing more."
ANSWER_GENERATION_TEMPLATE = "{instruction}\n\n### CONTEXT\n{context}\n\n### QUESTION:\n{question}\n\n### RESPONSE:"
ANSWER_GENERATION_DELIMITER = "### RESPONSE:"

# Response rewrite prompt using feedback critic
RESPONSE_REWRITE_INSTRUCTION = "Using the EVALUATION given, write an IMPROVED RESPONSE to the QUESTION under the CONTEXT provided"
RESPONSE_REWRITE_TEMPLATE = "{instruction}\n\n### CONTEXT\n{context}\n\n### QUESTION:\n{question}\n\n### RESPONSE:{answer}\n\n### EVALUATION:{evaluation}\n\n### IMPROVED RESPONSE:"
RESPONSE_REWRITE_DELIMITER = "### IMPROVED RESPONSE:"

# Post-process response
POST_PROCESS_RESPONSE_INSTRUCTION = '''
I have provided 10 examples of TriviaQA question-answer pairs below.
Examples:
### QUESTION: "Where in England was Dame Judi Dench born?",
### RESPONSE: "York",

### QUESTION": "From which country did Angola achieve independence in 1975?",
### RESPONSE: "Portugal",

### QUESTION": "Which city does David Soul come from?",
### RESPONSE: "Chicago",

### QUESTION": "Who won Super Bowl XX?",
### RESPONSE: "Chicago Bears",

### QUESTION": "Which was the first European country to abolish capital punishment?",
### RESPONSE: "Norway",

### QUESTION": "In which country did he widespread use of ISDN begin in 1988?",
### RESPONSE": "Japan",

### QUESTION": "What is Bruce Willis' real first name?",
### RESPONSE": "Walter",

### QUESTION": "Which William wrote the novel Lord Of The Flies?",
### RESPONSE": "Golding",

### QUESTION": "How is Joan Molinsky better known?",
### RESPONSE": "Joan Rivers",

### QUESTION": "In which branch of the arts is Patricia Neary famous?",
### RESPONSE": "Ballet",

Follow the style of the RESPONSE in the examples above to format the RESPONSE below. '''

POST_PROCESS_RESPONSE_TEMPLATE = "{instruction}\n\n### QUESTION:\n{question}\n\n### RESPONSE:{answer}\n\n### FORMATTED RESPONSE:"
POST_PROCESS_RESPONSE_DELIMITER = "### FORMATTED RESPONSE:"
