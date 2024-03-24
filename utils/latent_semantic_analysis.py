import math
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from transformers import AutoTokenizer

def truncate_text(
    text: str, llm_max_tokens: int, hf_tokenizer: AutoTokenizer, LANGUAGE="english"
) -> str:
    """
    Truncate_text using LSA summarization to reduce the text using the number of input token the LLM 
    accept. 

    Args:
        text (str): The text that need to be truncate
        llm_max_tokens (int): Maximum number of tokens the LLM support.
        hf_tokenizer (AutoTokenizer): HuggingFace tokenizer. Use to calculate number of tokens. 

    Retruns:
        Summary (str)
    """
    
    summarizer = LsaSummarizer()
    
    # How many toke the text have?
    num_tokens = len(hf_tokenizer.encode(text))

    if num_tokens > llm_max_tokens:
        print(f"Text is too long. Splitting into chunks of {llm_max_tokens} tokens")
        parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
        num_sentences = len(parser.document.sentences)
        avg_tokens_per_sentence = int(num_tokens / num_sentences)
        excess_tokens = num_tokens - llm_max_tokens
        num_sentences_to_summarize = num_sentences - (
            math.ceil(excess_tokens / avg_tokens_per_sentence)
        )

        print(f"Number of tokens: {num_tokens}.")
        print(f"Number of sentences: {num_sentences}.")
        print(f"Average tokens per sentence: {avg_tokens_per_sentence}.")
        print(f"Excess tokens: {excess_tokens}.")
        print(f"Number of sentences to summarize: {num_sentences_to_summarize}")

        summary = summarizer(parser.document, num_sentences_to_summarize)
        summary_text = "\n".join([sentence._text for sentence in summary])
        return truncate_text(summary_text, llm_max_tokens, hf_tokenizer)

    else:
        print("Text is short enough. No need to summarizing.")
        return text