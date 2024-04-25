# RAG-Critic: A Critic-based Framework for Correcting Hallucinations

This repository contains the code for RAG-Critic as part of the CS5260 Deep Learning and Neural Networks 2 module in NUS. The codebase is directly adapted from FastChat, and is structured as follows:

```
RAG-Critic
├── triviaqa_datasets
│   ├── TriviaQA
│   └── datasets.py
│
├── environment.yml
├── main.py
├── finetune.py
├── generate.py
├── scripts
│   ├── bootstrap_incorrect_responses.sh
│   ├── bootstrap_evaluation_generation.sh
│   ├── finetune_model.sh
│   ├── generate_response.sh
│   └── refine_response_with_critic.sh
│
└── utils
    ├── const.py
    ├── latent_semantic_analysis.py
    ├── prompt.py
    └── utils.py
```

`main.py` acts as the driver code for executing all functionality (dataset boostrapping and fine-tuning the critic model). The functionality and parameters for `main.py` are controlled by executing corresponding scripts under `scripts/`.

## Initializing the environment

1. Create the environment required for running the repository:

```
conda create -n rag-critic python=3.10
conda activate rag-critic
python -m pip install --upgrade pip # Ensure updated to pip 24.0
```

2. Install the `transformers` package:

```
python -m pip install transformers==4.37.2
python -m pip install sentence-transformers
```

**NOTE**: `transformers=4.38.1` seems have some class name inconsistencies, suggest to avoid using this version.

3. Install LangChain and related dependencies

```
python -m pip install --upgrade --quiet  langchain-openai tiktoken chromadb langchain
```

4. Install sumy for extractive summarization

```
python -m pip install sumy
```

5. Install datasets, bitsandbytes, perft, trl for Finetuning

```
python -m pip install datasets peft bitsandbytes trl
```

6. Install deepeval if using GEval for scoring zero-shot and critic-refined answers (optional).
```
python -m pip install deepeval
```

## Downloading the datasets

The primary dataset used in the repository is TriviaQA, which can be downloaded from the following link: https://nlp.cs.washington.edu/triviaqa/. Make sure to place the downloaded files for both RC and unfiltered under `triviaqa_datasets/TriviaQA/rc/` and `triviaqa_datasets/TriviaQA/unfiltered/` respectively.

## Using the code base

### 1. Generating Incorrect Responses

Execute the following command:

```
./scripts/bootstrap_incorrect_responses.sh
```

after adjusting the necessary paths and parameters.
- Our experiment generated 31,834 incorrect responses (web-train-incorrect-response.json).

### 2. Generating Evaluations for Correct and Incorrect Responses

Execute the following command:

```
./scripts/bootstrap_evaluation_generation.sh
```

after adjusting the necessary paths and parameters.
- Our experiment generated 27,784 evaluations, of which 13,892 are evaluations for correct responses, the other 13,892 are evaluations for incorrect responses (web_train_evaluation_generation.json).

### 3. Finetune Critic Model to predict Evaluations

Execute the following command:

```
./scripts/finetune_critic.sh
```

The weights of our finetuned model can be found [here](https://drive.google.com/drive/folders/1vum8GMIHifRrynfsEBBOppXEOyGXE8pV?usp=sharing)

after adjusting the necessary paths and parameters.

### 4. Generate Responses on Test Dataset

Execute the following command:

```
./scripts/generate_response.sh
```

after adjusting the necessary paths and parameters.
- Our experiment generated 7993 responses for TriviaQA's test set (web-dev-generated-response.json).

### 5. Refine Responses on Test Dataset using Critic Model

Execute the following command:

```
./scripts/refine_response_with_critic.sh
```

after adjusting the necessary paths and parameters.
- Our experiment refined 7993 responses for TriviaQA's test set (web-dev-refined-generated-response.json).

### 6. Post-processing refined responses (Optional)

### 7. Evaluation

