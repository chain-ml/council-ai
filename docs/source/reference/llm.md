# LLM

```{eval-rst}
.. autoclasstree:: council.llm
    :zoom:
    :full:
    :namespace: council

.. automodule:: council.llm
```

## Overview

The `council.llm` module provides a unified interface for interacting with various LLM providers, along with tools for handling responses, caching, logging and tracking consumptions metrics.

### LLMs

Create your LLM instance from YAML config file with {class}`~council.llm.LLMConfigObject` (see for different config examples).

Currently supported providers include: 

- OpenAI's GPT and o1 - {class}`~council.llm.OpenAILLM`
- Anthropic's Claude - {class}`~council.llm.AnthropicLLM`
- Google's Gemini - {class}`~council.llm.GeminiLLM`
- Microsoft's Azure - {class}`~council.llm.AzureLLM`
- Groq - {class}`~council.llm.GroqLLM`
- and local models with [ollama](https://ollama.com/) - {class}`~council.llm.OllamaLLM`

```{eval-rst}
.. testcode::

    from council.llm import get_llm_from_config

    # will adjust provider class automatically based on config file
    llm = get_llm_from_config("data/configs/llm-config-openai.yaml")
```

#### Making Requests and Managing Costs

Use `llm.post_chat_request()` method to interact with an LLM. The returned {class}`~council.llm.LLMResult` object contains LLM response as well as list of {class}`~council.contexts.Consumption` metrics associated with the call, including duration, token usage and costs.

```python
import dotenv
from council import LLMContext
from council.llm import LLMMessage, get_llm_from_config

llm = get_llm_from_config("data/configs/llm-config-openai.yaml")
result = llm.post_chat_request(
    LLMContext.empty(),
    messages=[LLMMessage.user_message("Hello world")]
)

print(result.first_choice)
# sample output:
# Hello! How can I assist you today?

for consumption in result.consumptions:
    print(consumption)
# sample output:
# gpt-4o-mini-2024-07-18 consumption: 1 call
# gpt-4o-mini-2024-07-18 consumption: 0.9347 second
# gpt-4o-mini-2024-07-18:prompt_tokens consumption: 9 token
# gpt-4o-mini-2024-07-18:completion_tokens consumption: 9 token
# gpt-4o-mini-2024-07-18:total_tokens consumption: 18 token
# gpt-4o-mini-2024-07-18:prompt_tokens_cost consumption: 1.3499e-06 USD
# gpt-4o-mini-2024-07-18:completion_tokens_cost consumption: 5.399e-06 USD
# gpt-4o-mini-2024-07-18:total_tokens_cost consumption: 6.7499e-06 USD
```

#### Anthropic Prompt Caching Support

For information about enabling Anthropic prompt caching, refer to {class}`~council.llm.LLMCacheControlData`.

### LLM Functions

LLM Functions provide structured ways to interact with LLMs including built-in response parsing, error handling and retries.

- See {class}`~council.llm.LLMFunction` for a code example
- Use {class}`~council.llm.LLMFunctionWithPrompt` to create an LLMFunction with {class}`~council.prompt.LLMPromptConfigObject`

### Response Parsers

Response parsers help automate the parsing of common response formats to use LLMFunctions conveniently:

- {class}`~council.llm.EchoResponseParser` for raw {class}`~council.llm.LLMResponse`
- {class}`~council.llm.StringResponseParser` for plain text
- {class}`~council.llm.CodeBlocksResponseParser` for code blocks
- {class}`~council.llm.YAMLBlockResponseParser` and {class}`~council.llm.YAMLResponseParser` for YAML
- {class}`~council.llm.JSONBlockResponseParser` and {class}`~council.llm.JSONResponseParser` for JSON

### LLM Middleware

Middleware components allow you to enhance LLM interactions by modifying requests and responses introducing custom logic, such as logging, caching, configuration updates, etc.

Core middlewares:

- Caching: {class}`~council.llm.LLMCachingMiddleware`
- Logging: 
  - Context logger: {class}`~council.llm.LLMLoggingMiddleware`
  - Files: {class}`~council.llm.LLMFileLoggingMiddleware` and {class}`~council.llm.LLMTimestampFileLoggingMiddleware`

Middleware management:

- {class}`~council.llm.LLMMiddlewareChain`
- {class}`~council.llm.LLMMiddleware`

### Fine-tuning and Batch API

See {class}`~council.prompt.LLMDatasetObject` for details on how to convert your YAML dataset into JSONL for fine-tuning and batch API.
Currently, the functionality is limited to generating JSONL files and does not include utilities for managing fine-tuning or batch job processes.

## Reference

```{eval-rst}
.. toctree::
    :maxdepth: 1
    :glob:

    llm/*
```
