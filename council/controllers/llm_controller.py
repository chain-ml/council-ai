from typing import List, Optional, Sequence, Tuple
from typing_extensions import TypeGuard

from council.chains import ChainBase
from council.contexts import AgentContext, ChatMessage
from council.llm import LLMBase, LLMMessage, MonitoredLLM
from council.utils import Option
from .controller_base import ControllerBase, ControllerException
from .execution_unit import ExecutionUnit
from council.llm.llm_answer import llm_property, LLMAnswer, LLMParsingException


class Specialist:
    def __init__(self, name: str, justification: str, instructions: str, score: int):
        self._instructions = instructions
        self._score = score
        self._name = name
        self._justification = justification

    @llm_property
    def name(self) -> str:
        """Specialist name to score for the task"""
        return self._name

    @llm_property
    def score(self) -> int:
        """Specialist relevance score"""
        return self._score

    @llm_property
    def instructions(self) -> str:
        """Specific instructions to give to this Specialist, or None if the Specialist is not supporting those."""
        return self._instructions

    @llm_property
    def justification(self):
        """Short and specific explanation of your score to this particular Specialist"""
        return self._justification


class LLMController(ControllerBase):
    """
    A controller that uses an LLM to decide the execution plan
    """

    _llm: MonitoredLLM

    def __init__(
        self,
        chains: Sequence[ChainBase],
        llm: LLMBase,
        response_threshold: float = 0.0,
        top_k: Optional[int] = None,
        parallelism: bool = False,
    ):
        """
        Initialize a new instance of an LLMController

        Parameters:
            llm (LLMBase): the instance of LLM to use
            response_threshold (float): a minimum threshold to select a response from its score
            top_k (int): maximum number of execution plan returned
            parallelism (bool): If true, Build a plan that will be executed in parallel
        """
        super().__init__(chains=chains, parallelism=parallelism)
        self._llm = self.register_monitor(MonitoredLLM("llm", llm))
        self._response_threshold = response_threshold
        if top_k is None:
            self._top_k = len(self._chains)
        else:
            self._top_k = min(top_k, len(self._chains))
        self._llm_controller_answer = LLMAnswer(Specialist)
        self._llm_system_message = self._build_system_message()
        self._retry = 3

    def _execute(self, context: AgentContext) -> List[ExecutionUnit]:
        retry = self._retry
        messages = self._build_llm_messages(context)
        new_messages = []
        while retry > 0:
            messages = messages + new_messages
            llm_result = self._llm.post_chat_request(context, messages)
            response = llm_result.first_choice
            context.logger.debug(f"llm response: {response}")
            try:
                retry -= 1
                plan = self._parse_response(context, response)
                plan.sort(key=lambda item: item[1], reverse=True)
                return [item[0] for item in plan if item[1] >= self._response_threshold][: self._top_k]
            except LLMParsingException as e:
                assistant_message = f"Your response is not correctly formatted:\n{response}"
                new_messages = self._handle_error(e, assistant_message, context)
            except Exception as e:
                assistant_message = f"Your response raised an exception:\n{response}"
                new_messages = self._handle_error(e, assistant_message, context)

        raise ControllerException("LLMController failed to execute.")

    @staticmethod
    def _handle_error(e: Exception, assistant_message: str, context: AgentContext) -> List[LLMMessage]:
        error = f"{e.__class__.__name__}: `{e}`"
        context.logger.warning(f"Exception occurred: {error}")
        return [LLMMessage.assistant_message(assistant_message), LLMMessage.user_message(f"Fix:\n{error}")]

    def _build_llm_messages(self, context: AgentContext) -> List[LLMMessage]:
        messages = [
            self._llm_system_message,
        ]
        message = context.chat_history.try_last_user_message
        if message.is_some():
            messages.append(LLMMessage.user_message(f"Score Specialists for:\n `{message.unwrap().message}`"))
        return messages

    def _build_system_message(self) -> LLMMessage:
        answer_choices = "\n".join(
            [f"name: {c.name};description: {c.description};{c.is_supporting_instructions}" for c in self._chains]
        )

        if self._top_k == 1:
            instruction = "Score only the most relevant and best Specialist."
        else:
            instruction = "Score all Specialists."
        task_description = [
            "# ROLE" "You are a knowledgeable expert responsible to fairly score Specialists.",
            "The score will reflect how relevant is a Specialist to solve or execute a user task.",
            "\n# INSTRUCTIONS",
            f"1. {instruction}",
            "2. Read carefully the user task and the Specialist description to score its relevance.",
            "3. Score from 0 (poor relevance or out of scope) to 10 (perfectly relevant).",
            "4. Ignore Specialist's name or its order in the list to give your score.",
            "5. If Specialist is supporting instructions, give any useful instructions to execute the user task.",
            "\n# FORMATTING",
            "1. Specialist list is precisely formatted as:",
            "name: {name};description: {description};{boolean indicating if Specialist is supporting instructions}",
            "2. Your response is precisely formatted as:",
            self._llm_controller_answer.to_prompt(),
            "\n# SPECIALISTS",
            answer_choices,
        ]
        return LLMMessage.system_message("\n".join(task_description))

    def _parse_response(self, context: AgentContext, response: str) -> List[Tuple[ExecutionUnit, int]]:
        parsed = [self._parse_line(context, line) for line in response.strip().splitlines()]
        filtered = [r.unwrap() for r in parsed if r.is_some()]
        if len(filtered) == 0:
            raise LLMParsingException(f"None of your scores could be parsed. Follow exactly formatting instructions.")

        if self._top_k > 1:
            actual_chains = [item[0].chain.name for item in filtered]
            missing_chains = [chain.name for chain in self._chains if chain.name not in actual_chains]
            if len(missing_chains) > 0:
                raise ControllerException(f"Missing scores for {missing_chains}. Follow exactly your instructions.")

        if len(filtered) != 1 and self._top_k == 1:
            raise ControllerException("You scored multiple Specialists. Score ONLY only the most relevant specialist.")

        return filtered

    def _parse_line(self, context: AgentContext, line: str) -> Option[Tuple[ExecutionUnit, int]]:
        if LLMAnswer.field_separator() not in line:
            return Option.none()

        cs: Optional[Specialist] = self._llm_controller_answer.to_object(line)
        if cs is not None:

            def typeguard_predicate(chain_base: ChainBase) -> TypeGuard[ChainBase]:
                return isinstance(chain_base, ChainBase) and chain_base.name.casefold() == cs.name.casefold()

            try:
                chain = next(filter(typeguard_predicate, self._chains))
                return Option.some((self._build_execution_unit(chain, context, cs.instructions, cs.score), cs.score))
            except StopIteration:
                context.logger.warning(f'message="no chain found with name `{cs.name}`"')
                raise ControllerException(f"The Specialist `{cs.name}` does not exist.")
        return Option.none()

    def _build_execution_unit(
        self, chain: ChainBase, context: AgentContext, instructions: str, score: int
    ) -> ExecutionUnit:
        return ExecutionUnit(
            chain,
            context.budget,
            initial_state=ChatMessage.chain(message=instructions) if chain.is_supporting_instructions else None,
            name=f"{chain.name};{score}",
            rank=self.default_execution_unit_rank,
        )
