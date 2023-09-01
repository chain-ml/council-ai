from typing import Optional

from council.monitors import Monitored
from ._execution_log import ExecutionLog
from ._execution_log_entry import ExecutionLogEntry


class ExecutionContext:
    _executionLog: ExecutionLog
    _entry: ExecutionLogEntry

    def __init__(self, execution_log: Optional[ExecutionLog] = None, path: str = ""):
        self._executionLog = execution_log or ExecutionLog()
        self._entry = self._executionLog.new_entry(path)

    def _new_path(self, monitored: Monitored):
        return monitored.name if self._entry.source == "" else f"{self._entry.source}/{monitored.name}"

    def new_for(self, monitored: Monitored) -> "ExecutionContext":
        return ExecutionContext(self._executionLog, self._new_path(monitored))

    @property
    def entry(self) -> ExecutionLogEntry:
        return self._entry
