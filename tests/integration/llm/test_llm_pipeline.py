from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional

import dotenv
from pydantic import BaseModel, Field

from council import OpenAILLM
from council.llm import YAMLBlockResponseParser
from council.llm.llm_function.llm_pipeline import (
    LLMProcessor,
    Processor,
    NaivePipelineProcessor,
    BacktrackingPipelineProcessor,
    ProcessorException,
)


class DatabaseQuestion(BaseModel):
    question: str
    table_name: str
    columns_with_types: List[Tuple[str, str]]

    def to_prompt(self) -> str:
        return "\n".join(
            [
                "Database Schema:",
                f"Table: {self.table_name}",
                *[f"{column}: {t}" for column, t in self.columns_with_types],
                "Please generate a SQL query to answer the following question:",
                self.question,
            ]
        )

    @classmethod
    def get_for_test(cls, question: str) -> DatabaseQuestion:
        return DatabaseQuestion(
            question=question,
            table_name="users",
            columns_with_types=[("id", "INTEGER"), ("name", "TEXT"), ("age", "INTEGER")],
        )


class SQLQuery(YAMLBlockResponseParser):
    feasible: bool = Field(..., description="Boolean, whether the query is feasible")
    query: str = Field(..., description="SQL query to answer the question OR explanation if the query is not feasible")

    def to_prompt(self) -> str:
        return "\n".join(["Explain the following query:", f"{self.query}"])


class SQLQueryExplained(YAMLBlockResponseParser):
    query: str = Field(..., description="The query itself without any modifications")
    analysis: str = Field(..., description="Analysis of the query quality and performance")


FakeTable = Dict[str, List[Any]]
FakeTableData: FakeTable = {"id": [1, 2, 3], "name": ["John", "Jane", "Doe"], "age": [28, 34, 29]}


class SQLQueryExecutor(Processor[SQLQuery, FakeTable]):
    def execute(self, obj: SQLQuery, exception: Optional[Exception] = None) -> FakeTable:
        if "limit" not in obj.query.lower():
            raise ProcessorException(
                input=obj.to_prompt(), message="Database is huge so query must contain a limit clause"
            )

        return FakeTableData


class SQLQueryExecutorWithTransfer(Processor[SQLQueryExplained, FakeTable]):
    def execute(self, obj: SQLQueryExplained, exception: Optional[Exception] = None) -> FakeTable:
        if "limit" not in obj.query.lower():
            raise ProcessorException(
                input=str(obj),  # TODO: need obj.to_response?
                message="Database is huge so query must contain a limit clause",
                transfer_to="SQLQuery",
            )

        return FakeTableData


dotenv.load_dotenv()
llm = OpenAILLM.from_env()


def get_question_to_query_proc() -> LLMProcessor[DatabaseQuestion, SQLQuery]:
    return LLMProcessor(llm, SQLQuery)


def get_query_to_explained_query_proc() -> LLMProcessor[SQLQuery, SQLQueryExplained]:
    return LLMProcessor(llm, SQLQueryExplained)


def get_execute_query_proc() -> Processor[SQLQuery, FakeTable]:
    return SQLQueryExecutor()


def get_execute_explained_query_proc() -> Processor[SQLQueryExplained, FakeTable]:
    return SQLQueryExecutorWithTransfer()


def get_naive_pipeline() -> NaivePipelineProcessor[DatabaseQuestion, SQLQueryExplained]:
    return NaivePipelineProcessor(
        [
            get_question_to_query_proc(),
            get_query_to_explained_query_proc(),
        ]
    )


def get_double_backtrack_pipeline() -> BacktrackingPipelineProcessor[DatabaseQuestion, FakeTable]:
    return BacktrackingPipelineProcessor(
        [
            get_question_to_query_proc(),
            get_execute_query_proc(),
        ]
    )


def get_triple_backtrack_pipeline() -> BacktrackingPipelineProcessor[DatabaseQuestion, FakeTable]:
    return BacktrackingPipelineProcessor(
        [
            get_question_to_query_proc(),
            get_query_to_explained_query_proc(),
            get_execute_explained_query_proc(),
        ]
    )


def test_question_to_query_proc():
    db_question = DatabaseQuestion.get_for_test("What is average salary?")
    proc = get_question_to_query_proc()
    query = proc.execute(db_question)
    assert isinstance(query, SQLQuery)
    assert not query.feasible
    assert len(proc.records) == 1
    assert proc.records[0].exception is None


def test_question_to_optimized_query_proc():
    db_question = DatabaseQuestion.get_for_test("How many users older than 30 are there?")
    question_to_query_proc = get_question_to_query_proc()
    query = question_to_query_proc.execute(db_question)
    assert isinstance(query, SQLQuery)
    assert query.feasible
    assert len(question_to_query_proc.records) == 1
    assert question_to_query_proc.records[0].exception is None

    query_to_explained_query_proc = get_query_to_explained_query_proc()
    optimized_query = query_to_explained_query_proc.execute(query)
    assert isinstance(optimized_query, SQLQueryExplained)
    assert len(query_to_explained_query_proc.records) == 1
    assert query_to_explained_query_proc.records[0].exception is None


def test_naive_pipeline():
    db_question = DatabaseQuestion.get_for_test("How many users older than 30 are there?")
    pipeline = get_naive_pipeline()
    explained_query = pipeline.execute(db_question)
    assert isinstance(explained_query, SQLQueryExplained)

    for proc in pipeline.processors:
        assert len(proc.records) == 1
        assert proc.records[0].exception is None


def test_double_backtracking_pipeline():
    db_question = DatabaseQuestion.get_for_test("How many users older than 30 are there?")
    pipeline = get_double_backtrack_pipeline()
    result = pipeline.execute(db_question)
    print(result)

    llm_proc = pipeline.processors[0]
    assert len(llm_proc.records) == 2
    assert llm_proc.records[0].exception.message == "Database is huge so query must contain a limit clause"
    assert llm_proc.records[1].exception is None


def test_triple_backtracking_pipeline():
    db_question = DatabaseQuestion.get_for_test("How many users older than 30 are there?")
    pipeline = get_triple_backtrack_pipeline()
    result = pipeline.execute(db_question)
    print(result)

    question_to_query_proc = pipeline.processors[0]
    assert len(question_to_query_proc.records) == 2
    assert (
        question_to_query_proc.records[0].exception.message == "Database is huge so query must contain a limit clause"
    )
    assert question_to_query_proc.records[1].exception is None

    query_to_optimized_query_proc = pipeline.processors[1]
    assert len(query_to_optimized_query_proc.records) == 2
    assert query_to_optimized_query_proc.records[0].exception is None
    assert query_to_optimized_query_proc.records[1].exception is None
