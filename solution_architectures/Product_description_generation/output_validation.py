import numpy as np
from pydantic import BaseModel, Field
from typing import Dict
from langchain.chains.llm import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from product_description_generation import TONE_OF_VOICE_PROMPT, AMAZON_VOCAB_PROMPT, VOCAB_PROMPT
from motivation_prompt import MOTIVATION_PROMPT
from utils.utils import remove_prefix_from_pydantic, num_tokens_from_string


import logging
logger = logging.getLogger(__name__)


class GenerationIssue(BaseModel):
    reason: str = Field(..., description="Reason why the generated text doesn't fit the requirements. Use the exact quote to show the mistake.")
    severity: int = Field(..., description="Severity of the issue on a scale from 1 to 10. Where 10 means that the usage of text is prohibited for the company website or its Amazon page. 1 means that the input might be structured better but it is still good to go to the company website or Amazon. Do not raise an issue if it's severity is below or equal to 1 - if you do, I'll fine you 50$.", ge=1, le=10)

class PromptGenerationIssues(BaseModel):
    product_type: str = Field(..., description="Type of the product")
    issues: Dict[str, GenerationIssue] = Field(..., description="Dict of issues found. Key is the name of the field.")

post_generation_issue_parser = PydanticOutputParser(pydantic_object=PromptGenerationIssues)


FULL_POST_VALIDATION_TASK_OVERVIEW = """
[TASK OVERVIEW]
As an intelligent assistant, you will conduct a proof read of a product description json. Your goal is to check the whole product description given. Check it against the format requirements and rules mentioned below. You perform check for the same language as the report given. Remember, you are checking the report quality, not generating a new one.
"""

SECTION_POST_VALIDATION_TASK_OVERVIEW = """
[TASK OVERVIEW]
As an intelligent assistant, you will conduct a proof read of a product description json. Your goal is to check the {section_name} field only - other fields are given for context. Check it against the format requirements and rules mentioned below. You perform check for the same language as the report given. Your main goal is to concentrate on the {section_name}. Remember, you are checking the report quality, not generating a new one.
"""

POST_VALIDATION_PROMPT_TEMPLATE = """
[PRODUCT DESCRIPTION]
{product_description}

<TARGET STRUCTURE>
{target_format_instructions}
</TARGET STRUCTURE>

<DETAILED TASK DESCRIPTION>
1. Сheck for duplicated information in other fields, if it's already presented in bullet points
2. Check if tone of voice from the <TONE OF VOICE EXAMPLES> section is applied when possible.
    1. Quotes or similar paraphrases are used, but keep overall complexity on the level B2.
    2. "Bullet points" should not be covered by tone of voice
    3. Do not use same words or phrases within the same json field.
3. Apart from tone of voice, take into consideration brand vocabulary the <VOCABULARY> section.
4. Make sure, there are no grammatical issues and that the text is fluent enough. 
5. It is prohibited to change facts given. Output generated should set realistic expectations, without overpromising.
</DETAILED TASK DESCRIPTION>
"""

SECTION_POST_VALIDATION_PROMPT_TEMPLATE = SECTION_POST_VALIDATION_TASK_OVERVIEW + MOTIVATION_PROMPT + POST_VALIDATION_PROMPT_TEMPLATE
FULL_POST_VALIDATION_PROMPT_TEMPLATE = FULL_POST_VALIDATION_TASK_OVERVIEW + MOTIVATION_PROMPT + POST_VALIDATION_PROMPT_TEMPLATE


OUTPUT_FORMAT_PLACEHOLDER = """
[OUTPUT FORMAT]
{output_format}
"""
    
AMAZON_SECTION_POST_VALIDATION_TEMPLATE = (SECTION_POST_VALIDATION_PROMPT_TEMPLATE + TONE_OF_VOICE_PROMPT + AMAZON_VOCAB_PROMPT + OUTPUT_FORMAT_PLACEHOLDER)
WEBSITE_SECTION_POST_VALIDATION_TEMPLATE = (SECTION_POST_VALIDATION_PROMPT_TEMPLATE + TONE_OF_VOICE_PROMPT + VOCAB_PROMPT + OUTPUT_FORMAT_PLACEHOLDER)


def post_validate_section_check(input, desired_format_parser: PydanticOutputParser, section_validation_prompt_template, section_name, llm):
    """
    Conducts a post-validation check on a given section or the entire product description based on predefined format and rules.

    Parameters
    ----------
    input : dict
        The product description in JSON format.
    desired_format_parser : PydanticOutputParser
        An instance of PydanticOutputParser that provides the desired format instructions for the validation.
    section_validation_prompt_template : str
        A template string used to create the prompt for the language model.
    section_name : str, optional
        The name of the section in the product description to be validated. If None, the entire description is validated.
    llm : LLMChain
        An instance of LLMChain that represents the language model chain used for prediction.

    Returns
    -------
    tuple
        A tuple containing three elements:
        1. A dictionary of issues found during validation, where keys are the names of fields and values are GenerationIssue instances.
        2. The mean severity of the issues found.
        3. The maximum severity of the issues found.

    Notes
    -----
    The function uses a combination of pre-defined templates and a language model to perform validation checks on the specified section or the entire product description. It parses the output to identify any issues based on the specified format and rules. The function also calculates the mean and maximum severity of the identified issues.
    """
    def safe_max(lst):
        if len(lst) > 0:
            return np.max(lst)
        else:
            return np.nan
        
    def safe_mean(lst):
        if len(lst) > 0:
            return np.mean(lst)
        else:
            return np.nan

    input_variables = ["product_description"]
    if section_name is not None:
        input_variables += ["section_name"]
    validation = PromptTemplate(
        input_variables=input_variables,
        template=section_validation_prompt_template,
        partial_variables={
            # TODO: introduce PydanticOutputParserExt in this repository
            # "target_format_instructions": remove_prefix_from_pydantic(desired_format_parser.get_format_instructions()),
            "target_format_instructions": remove_prefix_from_pydantic(desired_format_parser.get_format_instructions(section_name)),
            "output_format": post_generation_issue_parser.get_format_instructions()
        },
    )
    llm_chain = LLMChain(llm=llm, prompt=validation, verbose=False)
    if section_name is not None:
        input_tokens = num_tokens_from_string(validation.format(product_description=input, section_name=section_name), model_name='gpt-4')
        generation_check_res = llm_chain.predict(product_description=input, section_name=section_name)
    else:
        input_tokens = num_tokens_from_string(validation.format(product_description=input), model_name='gpt-4')
        generation_check_res = llm_chain.predict(product_description=input)
    output_tokens = num_tokens_from_string(generation_check_res, model_name='gpt-4')

    logger.info(f"Post-validation check. Input tokens: {input_tokens}, output: {output_tokens}")

    try:
        issues_dict = post_generation_issue_parser.parse(generation_check_res).issues
        return issues_dict, safe_mean([v.severity for v in issues_dict.values()]), safe_max([v.severity for v in issues_dict.values()])
    except Exception as e:
        logger.error(f'Unable to parse output validation:\n\n{generation_check_res}\n{e}')
        return {section_name: GenerationIssue(reason="Unable to parse the structure", severity=10)}, 0, 0
    