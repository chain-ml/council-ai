import abc
import logging
from typing import Any, Dict

from council.contexts import Budget, ChatMessage
from .scorer_exception import ScorerException

logger = logging.getLogger(__name__)


class ScorerBase(abc.ABC):
    """
    Base class for implementing a Scorer
    """

    def score(self, message: ChatMessage, budget: Budget) -> float:
        """
        Score the given message

        Parameters:
            message (ChatMessage): the message to be scored
            budget (Budget): the budget for scoring

        Returns:
            similarity score. The greater the value to higher the similarity

        Raises:
            ScorerException: an unexpected error occurs
        """
        try:
            return self._score(message, budget)
        except Exception:
            logging.exception('message="execution failed"')
            raise ScorerException

    @abc.abstractmethod
    def _score(self, message: ChatMessage, budget: Budget) -> float:
        """
        To be implemented with in derived classes with actual scoring logic
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the instance into a dictionary. May need to be overriden in derived classes
        """
        return {"type": self.__class__.__name__}
