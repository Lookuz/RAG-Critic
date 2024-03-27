# RAG-Critic: A Critic-based Framework for Correcting Hallucinations

This repository contains the code for RAG-Critic as part of the CS5260 Deep Learning and Neural Networks 2 module in NUS. The codebase is directly adapted from FastChat, and is structured as follows:

```
RAG-Critic
|   ├── FastChat
├── triviaqa_datasets
│   ├── NQ
│   ├── TriviaQA
│   └── datasets.py
│
├── environment.yml
├── main.py
├── finetune.py
├── scripts
│   ├── bootstrap_incorrect_responses.sh
│   ├── bootstrap_evaluation_generation.sh
│   └── finetune_model.sh
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
```

**NOTE**: `transformers=4.38.1` seems have some class name inconsistencies, suggest to avoid using this version.

<!-- 3. Install the FastChat dependencies:

```
cd FastChat
python -m pip install -e ".[model_worker,webui]"
``` -->

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

## Downloading the datasets

The primary dataset used in the repository is TriviaQA, which can be downloaded from the following link: https://nlp.cs.washington.edu/triviaqa/. Make sure to place the downloaded files for both RC and unfiltered under `triviaqa_datasets/TriviaQA/rc/` and `triviaqa_datasets/TriviaQA/unfiltered/` respectively.

## Using the code base

### 1. Generating False Responses

Execute the following command:

```
./scripts/bootstrap_incorrect_responses.sh
```

after adjusting the necessary paths and parameters.
