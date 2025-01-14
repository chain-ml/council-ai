from __future__ import annotations

from typing import Any, Dict, Final, Generic, List, Optional, Protocol, Sequence, Type, TypeVar, Union, cast

from council.llm.base import LLMBase, LLMMessage

from .llm_function import LLMFunction
from .llm_middleware import LLMMiddlewareChain
from .llm_response_parser import BaseModelResponseParser


class ProcessorException(Exception):
    """
    Exception raised during the execution of a Processor.
    Contains information about the input that caused the exception, the exception itself,
    and optionally name of a previous processor that should handle the exception.
    """

    def __init__(self, *, input: str, message: str, transfer_to: Optional[str] = None):
        self.input = input
        self.message = message
        self.transfer_to = transfer_to


class LLMProcessorInput(Protocol):
    """Input for a LLMProcessor."""

    def to_prompt(self) -> str:
        """Convert the object to a prompt string."""
        ...


T_Input = TypeVar("T_Input")
T_Output = TypeVar("T_Output")

# input must implement to_prompt()
T_LLMInput = TypeVar("T_LLMInput", bound=LLMProcessorInput)
# output must implement from_response() and to_response_template()
T_LLMOutput = TypeVar("T_LLMOutput", bound=BaseModelResponseParser)


class LLMProcessorRecord:
    """Record of a LLMProcessor execution."""

    def __init__(self, *, input: str, output: str, exception: Optional[ProcessorException] = None):
        self.input = input
        self.output = output
        self.exception = exception

    def set_exception(self, exception: ProcessorException) -> None:
        self.exception = exception

    def to_dict(self) -> Dict[str, str]:
        result = {
            "input": self.input,
            "output": self.output,
        }
        if self.exception is not None:
            result["exception"] = self.exception.message
        return result


class Processor(Generic[T_Input, T_Output]):
    """
    Base class for a Processor, transforming an input object into an output object.
    List of processors can be then used to create a PipelineProcessor.
    """

    def execute(self, obj: T_Input, exception: Optional[ProcessorException] = None) -> T_Output:
        # implement logic that transforms obj into T_Output
        raise NotImplementedError()


class LLMProcessor(Processor[T_LLMInput, T_LLMOutput]):
    """
    Processor that uses an LLM to convert an input object into an output object.
    Keeps track of records processed by this instance.
    """

    def __init__(
        self, llm: Union[LLMBase, LLMMiddlewareChain], output_obj_type: Type[T_LLMOutput], name: Optional[str] = None
    ):
        self._llm_middleware = LLMMiddlewareChain(llm) if not isinstance(llm, LLMMiddlewareChain) else llm
        self._output_obj_type = output_obj_type
        self.name = name or output_obj_type.__name__

        self._records: List[LLMProcessorRecord] = []
        self.PROMPT_TEMPLATE: Final[str] = "\n".join(["{input_obj_prompt}", "", "{response_template}"])

    @property
    def records(self) -> List[LLMProcessorRecord]:
        """List of all records processed by this instance."""
        return self._records

    @property
    def records_with_exceptions(self) -> List[LLMProcessorRecord]:
        """List of records processed by this instance that resulted in an exception."""
        return [record for record in self.records if record.exception is not None]

    def add_record(self, *, input_prompt: str, produced_output: str) -> None:
        self._records.append(LLMProcessorRecord(input=input_prompt, output=produced_output))

    def last_record(self) -> LLMProcessorRecord:
        # TODO: naive?
        if len(self.records) == 0:
            raise ValueError("No records found.")

        return self.records[-1]

    def execute(self, obj: T_LLMInput, exception: Optional[ProcessorException] = None) -> T_LLMOutput:
        system_prompt = self.PROMPT_TEMPLATE.format(
            input_obj_prompt=obj.to_prompt(),
            response_template=self._output_obj_type.to_response_template(),
        )
        messages = [LLMMessage.system_message(system_prompt)]
        if exception is not None:
            self.last_record().set_exception(exception)
            messages.extend([LLMMessage.assistant_message(exception.input), LLMMessage.user_message(exception.message)])

        llm_func: LLMFunction[T_LLMOutput] = LLMFunction(
            llm=self._llm_middleware,
            response_parser=self._output_obj_type.from_response,
            messages=messages,
        )
        llm_func_response = llm_func.execute_with_llm_response()
        self.add_record(input_prompt=system_prompt, produced_output=llm_func_response.llm_response.value)

        return llm_func_response.response


class PipelineProcessorBase(Generic[T_Input, T_Output]):
    """
    Base class for a PipelineProcessor, executing a sequence of Processors.
    """

    def __init__(self, processors: Sequence[Processor]):
        self.processors = list(processors)

    def execute(self, obj: T_Input) -> T_Output:
        raise NotImplementedError()


class NaivePipelineProcessor(PipelineProcessorBase[T_Input, T_Output]):
    """
    PipelineProcessor that executes processors in a linear order without any error handling.
    Each processor should be able to handle errors independently.

    .. mermaid::

        flowchart LR
            A(Processor A) --> B(Processor B)
            A --> A
            B --> B
    """

    def execute(self, obj: T_Input) -> T_Output:
        current_obj: Any = obj
        for processor in self.processors:
            current_obj = processor.execute(current_obj)
        return cast(T_Output, current_obj)


class BacktrackingPipelineProcessor(PipelineProcessorBase[T_Input, T_Output]):
    """
    PipelineProcessor that executes processors in a linear order backtracking errors.
    If a processor fails, the pipeline will backtrack to the previous processor (or specified in the exception)
    and try again.

    .. mermaid::

        flowchart LR
            A(Processor A) --> B(Processor B)
            A --> A
            B --> B
            B --> A
    """

    def __init__(self, processors: Sequence[Processor], max_backtracks: int = 3):
        super().__init__(processors)
        self.max_backtracks = max_backtracks

    @staticmethod
    def should_handle(processor: Processor, exception: Optional[ProcessorException] = None) -> bool:
        if exception is None:
            return True

        if not isinstance(processor, LLMProcessor):
            return False

        if exception.transfer_to is None:
            return True
        return exception.transfer_to == processor.name

    def execute(self, obj: T_Input) -> T_Output:
        index = 0
        inputs: List[T_Input] = [obj] * len(self.processors)
        previous_exception: Optional[ProcessorException] = None
        current_obj = obj
        backtrack_count = 0

        while index < len(self.processors):
            try:
                if not self.should_handle(self.processors[index], previous_exception):
                    index -= 1
                    continue
                current_obj = self.processors[index].execute(inputs[index], previous_exception)
                index += 1
                previous_exception = None  # reset exception after successful execution
                if index < len(self.processors):
                    inputs[index] = current_obj
            except ProcessorException as e:
                if backtrack_count >= self.max_backtracks:
                    raise e  # out of retry

                index -= 1
                backtrack_count += 1
                previous_exception = e

        return cast(T_Output, current_obj)
