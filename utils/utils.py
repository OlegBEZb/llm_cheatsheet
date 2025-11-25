"""
Some of the functions are taken from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
"""
import logging
from typing import List
from copy import deepcopy

import tiktoken
from langchain.prompts import PromptTemplate


def get_encoder(encoding_name, model_name):
    if encoding_name is not None:
        encoding = tiktoken.get_encoding(encoding_name)
    elif model_name is not None:
        encoding = tiktoken.encoding_for_model(model_name=model_name)
    else:
        raise ValueError("Either 'encoding_name' or 'model_name' have to be passed")
    return encoding


def num_tokens_from_string(string: str, encoding_name: str=None, model_name: str=None) -> int:
    """Returns the number of tokens in a text string."""
    encoding = get_encoder(encoding_name, model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens



def format_prompt_by_token_limit(input_texts: List[str], input_texts_argname: str, prompt: PromptTemplate, token_limit: int = 3000, model_name: str = 'gpt-4', **prompt_kwargs) -> List[str]:
    encoder = tiktoken.encoding_for_model(model_name)
    outliers_list_copy = deepcopy(input_texts)

    chunk_items = []
    for i in range(len(outliers_list_copy)):
        doc = outliers_list_copy.pop(0)
        logging.debug(f"Popped {i}th element")

        chunk_items.append(doc)
        current_prompt_input = "\n".join(chunk_items)
        prompt_kwargs[input_texts_argname] = current_prompt_input
        # TODO: use actual prompts input arg name
        formatted_prompt = prompt.format_prompt(**prompt_kwargs)
        current_prompt_len = len(encoder.encode(formatted_prompt.text))
        logging.debug(
            f"Trying chunk of {len(chunk_items)} size. Prompt len: {current_prompt_len}")
        if current_prompt_len > token_limit:
            logging.debug(
                f"Chunk of {len(chunk_items) - 1} size can't be extented")
            logging.debug(f"Inserting element back to 0 index")
            outliers_list_copy.insert(0, doc)
            chunk_items.pop(-1)
            logging.debug(f"Breaking")
            break

    logging.info(
        f"Returning chunk of {len(chunk_items)} items with remaining {len(outliers_list_copy)} items")
    return formatted_prompt, chunk_items, outliers_list_copy


# TODO: to replace chains with agents capable to check error codes
def refine_loop(initial_llm_chain, refine_llm_chain, input_docs):
    logging.info(f"Refine loop has {len(input_docs)} input docs")
    formatted_prompt, chunk_items, input_docs_list_remaining = format_prompt_by_token_limit(input_texts=input_docs,
                                                                                            input_texts_argname='reports_list',
                                                                                            prompt=initial_llm_chain.prompt,
                                                                                            token_limit=4000)
    logging.info(
        f"Initial chain to process a chunk of {len(chunk_items)} items. Remaining {len(input_docs_list_remaining)} items")
    res = initial_llm_chain.predict(reports_list=chunk_items)
    refine_steps = [res]
    while input_docs_list_remaining:
        logging.info(
            f"In while refinement loop. Remaining items to process: {len(input_docs_list_remaining)}")
        formatted_prompt, chunk_items, input_docs_list_remaining = format_prompt_by_token_limit(input_texts=input_docs_list_remaining,
                                                                                                input_texts_argname='reports_list',
                                                                                                prompt=refine_llm_chain.prompt,
                                                                                                token_limit=4000,
                                                                                                intermediate_result=clean_formatting_output(res))
        logging.info(
            f"Refine chain to process a chunk of {len(chunk_items)} items. Remaining {len(input_docs_list_remaining)} items")
        res = refine_llm_chain.predict(
            reports_list=chunk_items, intermediate_result=res)
        refine_steps.append(res)
    return refine_steps, res


def remove_prefix_from_pydantic(s):
    return s.split('Here is the output schema:\n')[-1]