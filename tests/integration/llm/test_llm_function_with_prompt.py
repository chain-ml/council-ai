import time
import unittest

import dotenv

from council.llm import AzureLLM, AnthropicLLM, EchoResponseParser, LLMFunctionWithPrompt
from council.prompt import LLMPromptConfigObject
from council.utils import OsEnviron
from tests import get_data_filename
from tests.integration.llm.test_llm_function import SQLResult
from tests.unit import LLMPrompts, XMLPrompts

DATASET_DESCRIPTION = """
# DATASET - nyc_airbnb
## Dataset Description
The dataset is the New York City Airbnb Open Data which includes information on Airbnb listings in NYC for the year 2019.
It provides data such as host id and name, geographical coordinates, room types, pricing, etc.

## Tables
### Table Name: NYC_2019

#### Table Description
Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present more unique, personalized way of experiencing the world. 
This data file includes all needed information to find out more about hosts, geographical availability, necessary metrics to make predictions and draw conclusions.

#### Columns
For each column, the name, data type and description are given as follow : {name}: {data type}: {description}
id: BIGINT: listing ID
name: TEXT: name of the listing
neighbourhood_group: TEXT: location
neighbourhood: TEXT: area
price: BIGINT: price in dollars
"""


class TestLlmFunctionWithPrompt(unittest.TestCase):
    """requires an Azure LLM model deployed"""

    def setUp(self) -> None:
        dotenv.load_dotenv()
        self.llm = AzureLLM.from_env()

        self.prompt_config_simple = LLMPromptConfigObject.from_yaml(get_data_filename(LLMPrompts.sql))
        self.prompt_config_template = LLMPromptConfigObject.from_yaml(get_data_filename(LLMPrompts.sql_template))
        self.prompt_config_large = LLMPromptConfigObject.from_yaml(get_data_filename(LLMPrompts.large))

        self.prompt_config_template_xml = LLMPromptConfigObject.from_yaml(get_data_filename(XMLPrompts.sql_template))

    def test_simple_prompt(self):
        llm_func = LLMFunctionWithPrompt(self.llm, SQLResult.from_response, self.prompt_config_simple)
        llm_function_response = llm_func.execute_with_llm_response()

        self.assertTrue(llm_function_response.duration > 0)
        self.assertTrue(len(llm_function_response.consumptions) > 0)
        sql_result = llm_function_response.response
        self.assertIsInstance(sql_result, SQLResult)
        print("", sql_result, sep="\n")

    def test_formatted_prompt(self):
        llm_func = LLMFunctionWithPrompt(
            self.llm,
            SQLResult.from_response,
            self.prompt_config_template,
            system_prompt_params={"dataset_description": DATASET_DESCRIPTION},
        )
        sql_result = llm_func.execute(user_prompt_params={"question": "Show me first 5 rows of the dataset"})
        self.assertIsInstance(sql_result, SQLResult)
        print("", sql_result, sep="\n")

    def test_formatted_xml_prompt(self):
        llm_func = LLMFunctionWithPrompt(
            self.llm,
            SQLResult.from_response,
            self.prompt_config_template_xml,
            system_prompt_params={"dataset_description": DATASET_DESCRIPTION},
        )
        sql_result = llm_func.execute(user_prompt_params={"question": "Show me first 5 rows of the dataset"})
        self.assertIsInstance(sql_result, SQLResult)
        print("", sql_result, sep="\n")

    def test_with_caching(self):
        with OsEnviron("ANTHROPIC_LLM_MODEL", "claude-3-haiku-20240307"):
            anthropic_llm = AnthropicLLM.from_env()

        llm_func_no_caching = LLMFunctionWithPrompt(
            anthropic_llm,
            EchoResponseParser.from_response,
            self.prompt_config_large,
            system_prompt_caching=False,
        )
        response = llm_func_no_caching.execute()

        assert all(not key.startswith("cache_") for key in response.result.raw_response["usage"])

        llm_func_caching = LLMFunctionWithPrompt(
            anthropic_llm,
            EchoResponseParser.from_response,
            self.prompt_config_large,
            system_prompt_caching=True,
        )
        response = llm_func_caching.execute()
        assert response.result.raw_response["usage"]["cache_creation_input_tokens"] > 0

        time.sleep(1)
        response = llm_func_caching.execute()
        assert response.result.raw_response["usage"]["cache_read_input_tokens"] > 0
