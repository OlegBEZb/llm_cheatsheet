from motivation_prompt import MOTIVATION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from utils.utils import num_tokens_from_string


import logging
logger = logging.getLogger(__name__)


def iter2str(i):
    return ', '.join(i)


# Phrases from brand book, slogans, etc
TONE_OF_VOICE_PROMPT = """
<TONE OF VOICE EXAMPLES>
1. 
2. 
</TONE OF VOICE EXAMPLES>
"""


GENERATION_PROMPT = """
<TASK OVERVIEW>
Your goal is to organize the information given about the product according to the structure and requirements.
</TASK OVERVIEW>

<INPUT DESCRIPTION>
Input is a text with all the information available about the product. The input may be a structured text with the fields names. Also input maybe completely unstructured and therefore you will have to identify the fields yourself.
</INPUT DESCRIPTION>

<INPUT>
{product_description}
</INPUT>

<DETAILED TASK DESCRIPTION>
1. Avoid duplicated information in other fields, if it's already presented in bullet points
2. Apply tone of voice from the <TONE OF VOICE EXAMPLES> section.
    1. Use quotes or similar paraphrases, but keep overall complexity on the level B2.
    2. "Bullet points" should not be covered by tone of voice
    3. Do not use same words or phrases within the same json field.
3. Apart from tone of voice, take into consideration the brand vocabulary the <VOCABULARY> section.
4. It is prohibited to change facts given. Output generated should set realistic expectations, without overpromising.
</DETAILED TASK DESCRIPTION>
"""

OUTPUT_PROMPT_PLACEHOLDER = """
<OUTPUT FORMAT>
{format_instructions}
</OUTPUT FORMAT>
"""

brand_vocabulary = ["<Brand>", "<Brand label>"]
good_to_use_words = ["Creative", "Experience", "Feel"]
avoid_to_use_words = ["Scientific", "Precise"]
avoid_to_use_words_amazon = ["special characters (TM, ®, €, ..., †, ‡, o, ¢, £, ¥, ©, ±, ~, â)"]
never_use_words = ["Buy", "Now!", "Save", "Deal", "Less", "Promo"]
never_use_words_amazon = ["Best seller", "return for a full refund", "lifetime guarantee", "free shipping", ]



VOCAB_PROMPT = f"""
<VOCABULARY>
Brand vocabulary: {iter2str(brand_vocabulary)}
Good to use: {iter2str(good_to_use_words)}
Avoid: {iter2str(avoid_to_use_words)}
Never use (there is a penalty of 50$ for using these words or phrases): {iter2str(never_use_words)}
</VOCABULARY>
"""

# extends the one above
AMAZON_VOCAB_PROMPT = f"""
<VOCABULARY>
Brand vocabulary: {iter2str(brand_vocabulary)}
Good to use: {iter2str(good_to_use_words)}
Avoid: {iter2str(avoid_to_use_words_amazon+avoid_to_use_words)}
Never use (there is a penalty of 50$ for using these words or phrases): {iter2str(never_use_words + never_use_words_amazon)}
</VOCABULARY>
"""
    
AMAZON_GENERATION_PROMPT = (GENERATION_PROMPT + TONE_OF_VOICE_PROMPT + AMAZON_VOCAB_PROMPT + OUTPUT_PROMPT_PLACEHOLDER + MOTIVATION_PROMPT)
WEBSITE_GENERATION_PROMPT = (GENERATION_PROMPT + TONE_OF_VOICE_PROMPT + VOCAB_PROMPT + OUTPUT_PROMPT_PLACEHOLDER + MOTIVATION_PROMPT)


def generate_description(raw_product_descr: str, prompt: str, product_structure_parser: PydanticOutputParser, llm):
    # prompt is either AMAZON_GENERATION_PROMPT or WEBSITE_GENERATION_PROMPT in this case
    prompt = PromptTemplate(input_variables=["product_description", "format_instructions"], template=prompt)
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    logger.info(f"Generating description from {num_tokens_from_string(prompt.format(product_description=raw_product_descr), model_name='gpt-4')} input tokens")
    full_product_description_generated = llm_chain.predict(product_description=raw_product_descr, format_instructions=product_structure_parser.get_format_instructions())
    logger.info(f"Raw generation {num_tokens_from_string(full_product_description_generated, model_name='gpt-4')}\n{full_product_description_generated}")
    product_description_generated_parsed = product_structure_parser.parse(full_product_description_generated)
    return product_description_generated_parsed