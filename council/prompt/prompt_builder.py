import logging
from typing import Any, List, Optional

from jinja2 import Template

from council.core import ChainContext, ChatMessageKind

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    A class for building prompts using a Jinja2 template and optional instructions.

    Args:
        t (str): The Jinja2 template string for the prompt.
        instructions (Optional[List[str]]): Optional instructions to be appended to the prompt.

    Attributes:
        _template (Template): The Jinja2 template object.
        _instructions (str): The instructions to be appended to the prompt.

    Methods:
        apply(context: ChainContext) -> str:
            Builds and returns the prompt by rendering the template and appending instructions.

    """

    def __init__(self, t: str, instructions: Optional[List[str]] = None):
        """
        Initializes a PromptBuilder instance.

        Args:
            t (str): The Jinja2 template string for the prompt.
            instructions (Optional[List[str]]): Optional instructions to be appended to the prompt.
        """

        self._template = Template(t)
        if instructions is not None and len(instructions) > 0:
            self._instructions = "\n# Instructions: "
            self._instructions += "\n".join(instructions)
        else:
            self._instructions = ""

    def apply(self, context: ChainContext) -> str:
        """
        Builds and returns the prompt by rendering the template and appending instructions.

        Args:
            context (ChainContext): The context object containing the necessary data for rendering the template.

        Returns:
            str: The generated prompt string.

        """

        template_context = {
            "chat_history": self.__build_chat_history(context),
            "chain_history": self.__build_chain_history(context),
        }

        prompt = self._template.render(template_context)
        prompt += self._instructions
        return prompt

    @staticmethod
    def __build_chat_history(context: ChainContext) -> dict[str, Any]:
        last_message = context.chatHistory.last_message()
        last_user_message = context.chatHistory.last_user_message()
        last_agent_message = context.chatHistory.last_agent_message()

        return {
            "agent": {
                "messages": [
                    msg.message for msg in context.chatHistory.messages if msg.is_of_kind(ChatMessageKind.Agent)
                ],
                "last_message": last_agent_message.map_or(lambda m: m.message, ""),
            },
            "user": {
                "messages": [
                    msg.message for msg in context.chatHistory.messages if msg.is_of_kind(ChatMessageKind.User)
                ],
                "last_message": last_user_message.map_or(lambda m: m.message, ""),
            },
            "messages": [msg.message for msg in context.chatHistory.messages],
            "last_message": last_message.map_or(lambda m: m.message, ""),
        }

    @staticmethod
    def __build_chain_history(context: ChainContext) -> dict[str, Any]:
        if len(context.chainHistory) == 0:
            return {
                "messages": [],
                "last_message": "",
            }

        last_message = context.current.last_message()
        return {
            "messages": [msg.message for msg in context.current.messages],
            "last_message": last_message.map_or(lambda m: m.message, ""),
        }
