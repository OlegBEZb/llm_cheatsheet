import os
import sys
import numpy as np
from pydantic import BaseModel, Field
from typing import Union, Dict, Tuple
from langchain.chains.llm import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from motivation_prompt import MOTIVATION_PROMPT

script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(script_path)
project_root = os.path.dirname(scripts_dir)
sys.path.append(project_root)

from utils.utils import remove_prefix_from_pydantic, num_tokens_from_string

import logging
logger = logging.getLogger(__name__)


class InputIssue(BaseModel):
    """
    A model representing an issue with the input information provided for generating a structured product description.
    """
    reason: str = Field(..., description="Reason why the input information is not sufficient or not valid.")
    severity: int = Field(..., description="Severity of the issue on a scale from 1 to 10. Where 10 means that the usage of this information is completely impossible for business needs. 1 means that the input might be structured better but further processing of this input is not under risk", ge=1, le=10)

class PromptInputIssues(BaseModel):
    """
    A model representing the issues found in the input product description.
    """
    product_type: str = Field(..., description="Type of the product from Page Title, one of [machines, capsules, packs]")
    issues: Dict[str, InputIssue] = Field(..., description="Dict of issues found. Key is the name of the field. In case of nested fields, wrap all the underlying elements of this field in one issue.")

input_issue_parser = PydanticOutputParser(pydantic_object=PromptInputIssues)


input_check_prompt = ("""
<TASK OVERVIEW>
Your goal is to check <INPUT PRODUCT DESCRIPTION> that it has enough information to generate a structured product description as described in <TARGET STRUCTURE>. You can use only information given in the <INPUT PRODUCT DESCRIPTION> - do not fill the missed fields from the <TARGET STRUCTURE>. <TARGET STRUCTURE> is given as a template of a structure. The input doesn't have to be structured like in the <TARGET STRUCTURE>, names of the fields may be completely different and may be organised differenly. However, input must not contain contradictions - if there are facts which contradict to each other, this is an issue. Your goal, again, is to decide if a new potential object of <TARGET STRUCTURE> structure may be created using information from <INPUT PRODUCT DESCRIPTION>.
</TASK OVERVIEW>

<INPUT PRODUCT DESCRIPTION>
{input_description}
</INPUT PRODUCT DESCRIPTION>

<TARGET STRUCTURE>
{target_format_instructions}
</TARGET STRUCTURE>

<OUTPUT FORMAT>
{output_format}
</OUTPUT FORMAT>
"""
+ MOTIVATION_PROMPT
)


def validate_input(raw_input: str, desired_format: Union[str, PydanticOutputParser], llm) -> Tuple[Dict[str, InputIssue], float, float]:
    """
    Validates the input information against a desired format and returns the issues found along with their severity.

    Parameters
    ----------
    raw_input : str
        The raw input product description to be validated.
    desired_format : Union[str, PydanticOutputParser]
        The desired format or structure that the input should conform to, or a PydanticOutputParser object containing the format instructions.
    llm : LLMChain
        The language model chain to be used for validation.

    Returns
    -------
    Tuple[Dict[str, InputIssue], float, float]
        A tuple containing a dictionary of issues found, the average severity of the issues, and the maximum severity of the issues.

    Raises
    ------
    Exception
        If the input validation result cannot be parsed.
    """
    if isinstance(desired_format, PydanticOutputParser):
        desired_format = remove_prefix_from_pydantic(desired_format.get_format_instructions())

    input_check_prompt_template = PromptTemplate(
        input_variables=["input_description"],
        template=input_check_prompt,
        partial_variables={
            "target_format_instructions": desired_format,
            "output_format": input_issue_parser.get_format_instructions()
        },
    )
    input_tokens = num_tokens_from_string(input_check_prompt_template.format(input_description=raw_input), model_name='gpt-4')
    
    llm_chain = LLMChain(llm=llm, prompt=input_check_prompt_template, verbose=False)
    logger.info(f"Pre-validation check. Input tokens: {input_tokens}")
    input_check_result = llm_chain.predict(input_description=raw_input)
    logger.info(f"Raw input check result\n{input_check_result}")
    
    output_tokens = num_tokens_from_string(input_check_result, model_name='gpt-4')
    logger.info(f"Pre-validation check. Output tokens: {output_tokens}")
    
    try:
        issues_dict = input_issue_parser.parse(input_check_result).issues
        return issues_dict, np.mean([v.severity for v in issues_dict.values()]), np.max([v.severity for v in issues_dict.values()])
    except Exception as e:
        logger.error(f'Unable to parse input validation:\n\n{input_check_result}\n{e}')
        return {}, 0, 0