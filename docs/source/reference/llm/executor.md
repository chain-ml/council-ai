# ParallelExecutor

```{eval-rst}
.. autoclass:: council.llm.ParallelExecutor
```

## Example

Here's an example of how to use `ParallelExecutor` to run multiple LLM requests in parallel for evaluation use case.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from dotenv import load_dotenv
from pydantic import Field

# !pip install council-ai==0.0.29

from council.llm import AnthropicLLM, JSONResponseParser, LLMBase, LLMFunction, OpenAILLM, ParallelExecutor


class EvaluationResponse(JSONResponseParser):
    reasoning: str = Field(..., description="Concise reasoning behind the evaluation")
    score: float = Field(..., ge=0, le=1, description="The score of the evaluation, between 0 and 1")


@dataclass
class AggregatedEvaluationResponse:
    evaluations: List[EvaluationResponse] = Field(..., description="The evaluations of the responses")
    average_score: float = Field(..., description="The average score of the evaluations")

    @classmethod
    def from_evaluations(cls, evaluations: Sequence[EvaluationResponse]) -> AggregatedEvaluationResponse:
        return cls(
            evaluations=list(evaluations),
            average_score=sum(e.score for e in evaluations) / len(evaluations),
        )


SYSTEM_PROMPT_TEMPLATE = """
You are a helpful assistant that evaluates the quality of a response.
Evaluate if the given response from AI assistant is helpful and accurate.

# Response format
{response_template}
"""

SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(response_template=EvaluationResponse.to_response_template())


def get_eval_llm_function(llm: LLMBase):
    return LLMFunction(
        llm=llm,
        response_parser=EvaluationResponse.from_response,
        system_message=SYSTEM_PROMPT,
    )


load_dotenv()

openai_llm_function = get_eval_llm_function(OpenAILLM.from_env())
anthropic_llm_function = get_eval_llm_function(AnthropicLLM.from_env())

# using different LLMs for this example, could be a single function and n>1
executor: ParallelExecutor[EvaluationResponse, AggregatedEvaluationResponse] = ParallelExecutor(
    [openai_llm_function.execute, anthropic_llm_function.execute],
    reduce=AggregatedEvaluationResponse.from_evaluations,
)

response = executor.execute_and_reduce("Thomas Jefferson was the 12th president of the United States.")
print(type(response))  # AggregatedEvaluationResponse
print(response)
# AggregatedEvaluationResponse(
#  evaluations=[
#   EvaluationResponse(reasoning='The given statement is inaccurate. Thomas Jefferson was the 3rd president of the United States, not the 12th.', score=0.2),
#   EvaluationResponse(reasoning='The statement is inaccurate. Thomas Jefferson was actually the 3rd president of the United States, not the 12th.', score=0.0)
#  ],
#  average_score=0.1)
```
