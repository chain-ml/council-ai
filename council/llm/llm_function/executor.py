from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Generic, List, Optional, Sequence, TypeVar, Union

T = TypeVar("T")
R = TypeVar("R")

Execute = Callable[..., T]
Reduce = Callable[[Sequence[T]], R]


class ParallelExecutor(Generic[T, R]):
    """
    Executes one or more functions (e.g. LLMFunction.execute) in parallel
    and aggregates the results according to a reduce function.

    Args:
        executes (Union[Execute, Sequence[Execute]]): Single function or sequence of functions to execute in parallel
        reduce (Optional[Reduce]): Function to aggregate sequence of responses into a response of potentially different type.
            Defaults to returning the first result.
        n (int): Number of times to execute if single function provided. Ignored if sequence provided.

    Raises:
        ValueError: If _executes sequence is empty or contains non-callable items

    .. mermaid::

        flowchart LR
            E1(Execute 1) --> R(Reduce)
            E3(Execute ...) --> R
            E4(Execute N) --> R
    """

    def __init__(
        self, executes: Union[Execute, Sequence[Execute]], reduce: Optional[Reduce] = None, n: int = 1
    ) -> None:
        self._executes: List[Execute] = self._validate_executes(executes, n)
        self._reduce: Reduce = reduce if reduce is not None else lambda results: results[0]

    @staticmethod
    def _validate_executes(executes: Union[Execute, Sequence[Execute]], n: int) -> List[Execute]:
        if not isinstance(executes, Sequence):
            if not callable(executes):
                raise ValueError("`executes` must be callable or sequence of callables")
            return [executes] * n
        else:
            if not executes:
                raise ValueError("`executes` sequence cannot be empty")
            if not all(callable(ex) for ex in executes):
                raise ValueError("All items in `executes` must be callable")
            return list(executes)

    def execute(self, *args, **kwargs) -> List[T]:
        """
        Execute functions in parallel and return all results.

        Args:
            *args: Positional arguments to pass to the execute functions
            **kwargs: Keyword arguments to pass to the execute functions

        Returns:
            List[T]: List of all parallel execution results
        """
        results: List[T] = []

        with ThreadPoolExecutor(max_workers=len(self._executes)) as executor:
            future_to_runner = {executor.submit(execute, *args, **kwargs): execute for execute in self._executes}

            for future in as_completed(future_to_runner):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    for f in future_to_runner:
                        f.cancel()
                    raise e

        return results

    def reduce(self, results: Sequence[T]) -> R:
        return self._reduce(results)

    def execute_and_reduce(self, *args, **kwargs) -> R:
        """
        Execute the function(s) in parallel and aggregate their results.

        Args:
            *args: Positional arguments to pass to the execute functions
            **kwargs: Keyword arguments to pass to the execute functions

        Returns:
            R: Result from reducing all parallel executions, potentially of different type than inputs
        """
        results = self.execute(*args, **kwargs)
        return self.reduce(results)
