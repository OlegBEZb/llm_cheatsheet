from typing import List
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from utils.utils import num_tokens_from_string


import logging
logger = logging.getLogger(__name__)


GENERATION_PROMPT = """
<TASK OVERVIEW>
Your goal is to generate a Python code for Pydantic V2 structure according to a semi-structured input. Do not use any validators. Just define structure. If you use validators, I'll fine you 100$.
</TASK OVERVIEW>

<INPUT DESCRIPTION>
Input is a text with structure described in human language. This may be a json definition created with pydantic PydanticOutputParser get_format_instructions function. This may also be a completely unstructured business level human language.
</INPUT DESCRIPTION>

<INPUT>
{structure_description}
</INPUT>

<DETAILED TASK DESCRIPTION>
1. The code must contain necessary python libraries imports and Pydantic classes
2. The cost must be safe to execute
3. Sometimes users define desired lengths in words but Pydantic works only with characters lengths. Assume, that one word is about 5 characters.
4. Usually there are some comments for the whole class rather than for separate fields. Keep such class comments as they are useful.
5. Generated classed must be named BulletPointsStructure and ProductStructure regardless of the names given in the input
</DETAILED TASK DESCRIPTION>
"""

FEW_SHOTS = """
<GREAT SIMPLIFIED EXAMPLE №1>
from pydantic import BaseModel, Field
class BulletPointsStructure(BaseModel):
    \"""
    Each bullet point should be easy to read and maximum 1 line on a mobile device. Minimum 4 words. There must not be a '.' at the end of any bullet point. If you end a bullet point with '.', I'll fine you 50$.
    \"""
    bullet_point_1: str = Field(min_length=30, description="", examples=["", ""])
    bullet_point_2: str = Field(max_length=255, description="", examples=["", ""])
=====
from pydantic import BaseModel, Field
class ProductStructure(BaseModel):
    URL: str = Field(description="URL, do not change it. If empty, generate an empty string")
    H1: str = Field(description="Main header aka H1 Title")
    bullet_points: BulletPointsStructure
</GREAT SIMPLIFIED EXAMPLE №1>
"""

OUTPUT_PROMPT_PLACEHOLDER = """
<OUTPUT FORMAT>
Python code with imports and Pydantic V2 classes. Will be further executed with 'exec' command. 

Output must be organised as two classes: bullet points class and a common class. Do not use any wrappings - just code (no need for ```python etc). Both pieces of code have to be separated like in a pseudocode below:
imports for bullet points class
BulletPointsStructure 
=====
imports for common structure class even if they are the same as the code is executed separately
ProductStructure 
</OUTPUT FORMAT>
"""
    
STRUCTURE_GENERATION_PROMPT = (GENERATION_PROMPT + FEW_SHOTS + OUTPUT_PROMPT_PLACEHOLDER)


structure_generation_prompt_template = PromptTemplate(
        input_variables=["structure_description"],
        template=STRUCTURE_GENERATION_PROMPT,
    )

def extract_pydantic_from_text(raw_structure: str, llm) -> List[str]:
    """Generates Python code for BulletPointsStructure and ProductStructure
    
    Example of usage:
    class_definitions = extract_pydantic_from_text(raw_structure)
    for class_name, class_code in class_definitions:
        logger.info(f"Executing the code below to initialise a structure class '{class_name}'\n```\n{class_code}\n```\n\n")
        exec(class_code)
        globals()[class_name] = locals()[class_name]
    product_structure_parser = PydanticOutputParserExt(pydantic_object=ProductStructure)
    bullet_point_structure_parser = PydanticOutputParserExt(pydantic_object=BulletPointsStructure)
    """
    llm_chain = LLMChain(llm=llm, prompt=structure_generation_prompt_template, verbose=True)
    generation = llm_chain.predict(structure_description=raw_structure)
    logger.info(f"Extracted Pydantic from text. Code generated size: {num_tokens_from_string(generation, model_name='gpt-4')} tokens")
    return zip(['BulletPointsStructure', 'ProductStructure'], generation.replace('```python', '').replace('```', '').split('====='))