from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import List, Sequence

import dotenv
from pydantic import Field

from council.llm import OpenAILLM, AnthropicLLM, LLMFunction, JSONResponseParser, ParallelExecutor
from council.utils import OsEnviron


class EvaluationResponse(JSONResponseParser):
    reasoning: str = Field(..., description="Concise reasoning behind the evaluation")
    score: float = Field(..., ge=0, le=1, description="The score of the evaluation, between 0 and 1")


@dataclass
class AggregatedEvaluationResponse:
    evaluations: List[EvaluationResponse]
    average_score: float

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


class TestParallelExecutor(unittest.TestCase):
    def setUp(self) -> None:
        dotenv.load_dotenv()
        with (
            OsEnviron("OPENAI_LLM_MODEL", "gpt-4o-mini"),
            OsEnviron("OPENAI_LLM_TEMPERATURE", "1.5"),
            OsEnviron("ANTHROPIC_LLM_MODEL", "claude-3-haiku-20240307"),
        ):

            self.openai_llm = OpenAILLM.from_env()
            self.anthropic_llm = AnthropicLLM.from_env()

    @staticmethod
    def get_eval_llm_function(llm) -> LLMFunction[EvaluationResponse]:
        return LLMFunction(
            llm=llm,
            response_parser=EvaluationResponse.from_response,
            system_message=SYSTEM_PROMPT,
        )

    def test_with_multiple_executes(self):
        openai_llm_function = self.get_eval_llm_function(self.openai_llm)
        anthropic_llm_function = self.get_eval_llm_function(self.anthropic_llm)

        executor = ParallelExecutor(
            [openai_llm_function.execute, anthropic_llm_function.execute],
            reduce=AggregatedEvaluationResponse.from_evaluations,
        )

        results = executor.execute("Thomas Jefferson was the 12th president of the United States.")

        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, EvaluationResponse)

        aggregated = executor.reduce(results)
        self.assertIsInstance(aggregated, AggregatedEvaluationResponse)
        self.assertEqual(len(aggregated.evaluations), 2)

    def test_with_n(self):
        llm_func = self.get_eval_llm_function(self.openai_llm)
        executor = ParallelExecutor(llm_func.execute, reduce=AggregatedEvaluationResponse.from_evaluations, n=3)
        results = executor.execute("Thomas Jefferson was the 12th president of the United States.")

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, EvaluationResponse)

        aggregated = executor.reduce(results)
        self.assertIsInstance(aggregated, AggregatedEvaluationResponse)
        self.assertEqual(len(aggregated.evaluations), 3)
