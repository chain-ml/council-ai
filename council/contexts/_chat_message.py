"""

Module _chat_message

This module implements data structures to represent and manipulate chat messages with
varied attributes such as kind, content, associated data, and source. It includes classes for defining
different types of messages (e.g., user, agent, skill, chain), performing operations on them, and
representing them with both scores and errors.

Classes:
	ChatMessageKind(Enum):
		An enumeration of different kinds of chat messages including user, agent, chain, and skill messages.
	
	ChatMessage:
		Represents the details of a chat message including its content, type, any associated data, and its source.
		This class also provides methods to generate predefined types of messages, and properties to access
		the message attributes.

		Additionally, the class includes utility functions to represent the message as a
		string or a dictionary.

		Attributes:
			_message (str): The text content of the chat message.
			_kind (ChatMessageKind): The categorical type of the chat message.
			_data (Any): Any additional data associated with the chat message.
			_is_error (bool): Flag indicating whether the message represents an error.
			_source (str): A string representing the source of the chat message.

		Methods:
			static agent(cls, message, data=None, source='', is_error=False): Constructs a ChatMessage of kind 'Agent'.
			static user(cls, message, data=None, source='', is_error=False): Constructs a ChatMessage of kind 'User'.
			static skill(cls, message, data=None, source='', is_error=False): Constructs a ChatMessage of kind 'Skill'.
			static chain(cls, message, data=None, source='', is_error=False): Constructs a ChatMessage of kind 'Chain'.

	ScoredChatMessage:
		Encapsulates a ChatMessage with an associated score, allowing for the comparison and sorting
		of messages based on their relevance or significance.

		Attributes:
			message (ChatMessage): The underlying ChatMessage object.
			score (float): A numerical score associated with the ChatMessage instance.

		Methods:
			__gt__: Comparison method for 'greater than' relation between two ScoredChatMessage instances.
			__lt__: Comparison method for 'less than' relation between two ScoredChatMessage instances.
			__ge__: Comparison method for 'greater than or equal to' relation.
			__le__: Comparison method for 'less than or equal to' relation.

Functions:
	Properties and utility methods to access and manage the attributes of ChatMessage and
	ScoredChatMessage instances, and to check their types and sources.


"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict


class ChatMessageKind(str, Enum):
    """
    A subclass of `str` and `Enum`, representing the different types of chat message senders within a chat system.
    
    Attributes:
        User (str):
             A constant representing a chat message originating from the user.
        Agent (str):
             A constant representing a chat message originating from the agent or customer support representative.
        Chain (str):
             A constant representing a chat message that is part of a chain of messages.
        Skill (str):
             A constant representing a chat message generated by a specific skill or functionality within the chat system.
    
    Methods:
        __str__(self):
             Overrides the string representation method to return the Enum value of the chat message kind.
        

    """

    User = "USER"
    """
    Represents a chat message from the user.
    """

    Agent = "AGENT"
    """
    Represents a chat message from the agent or customer support representative.
    """

    Chain = "CHAIN"
    """
    Represents a chat message from a chain
    """

    Skill = "SKILL"
    """
    Represents a chat message generated by a specific skill or functionality within the chat system.
    """

    def __str__(self):
        """
        Special method to return the string representation of the object.
        This method returns a string that consists of the value attribute of the object.
        It overrides the default implementation to provide a more meaningful string
        representation, which can be useful for debugging and logging purposes.
        
        Returns:
            (str):
                 A string representation of the object's value attribute.

        """
        return f"{self.value}"


class ChatMessage:
    """
    A class representing a chat message and its associated metadata.
    
    Attributes:
        _message (str):
             The primary text content of the chat message.
        _kind (ChatMessageKind):
             The enumerated type representing the kind/source of the message.
        _data (Any):
             Additional data related to the chat message.
        _is_error (bool):
             Flag indicating if the message represents an error state.
        _source (str):
             The origin identifier of the message, typically used to track the source of the message in a system.
    
    Methods:
        __init__:
             Constructs a new ChatMessage instance with the provided attributes.
        agent:
             Static method to create a ChatMessage instance of kind `Agent`.
        user:
             Static method to create a ChatMessage instance of kind `User`.
        skill:
             Static method to create a ChatMessage instance of kind `Skill`.
        chain:
             Static method to create a ChatMessage instance of kind `Chain`.
        message:
             Property that returns the chat message text.
        kind:
             Property that returns the kind of the message.
        is_kind_skill:
             Property that checks if the message kind is `Skill`.
        is_kind_agent:
             Property that checks if the message kind is `Agent`.
        is_kind_chain:
             Property that checks if the message kind is `Chain`.
        is_kind_user:
             Property that checks if the message kind is `User`.
        data:
             Property that returns the associated data of the message.
        source:
             Property that returns the source of the message.
        is_ok:
             Property that checks if the message does not represent an error.
        is_error:
             Property that checks if the message represents an error.
        is_of_kind:
             Method that checks if the message is of a specified kind.
        is_from_source:
             Method that checks if the message originates from a specified source.
        __str__:
             Defines the string representation of the ChatMessage instance.
        to_string:
             Returns a truncated version of the message's string representation, with a specified maximum length.
        to_dict:
             Returns a dictionary representation of the ChatMessage with its primary attributes.

    """

    _message: str
    _kind: ChatMessageKind
    _data: Any
    _is_error: bool
    _source: str

    def __init__(self, message: str, kind: ChatMessageKind, data: Any = None, source: str = "", is_error: bool = False):
        """
        Initializes a new instance of the chat message class.
        
        Args:
            message (str):
                 The text message content.
            kind (ChatMessageKind):
                 An enum representing the kind of message (e.g., command, response, etc).
            data (Any, optional):
                 Additional data related to the message. Defaults to None.
            source (str, optional):
                 The source identifier from which the message originated. Defaults to an empty string.
            is_error (bool, optional):
                 Flag indicating if the message is an error. Defaults to False.
        
        Attributes:
            _message (str):
                 The text of the chat message.
            _kind (ChatMessageKind):
                 The type/category of the message.
            _data (Any):
                 Additional information or payload accompanying the message.
            _source (str):
                 The identifier of the message source.
            _is_error (bool):
                 Indicates whether the message denotes an error condition.
            

        """
        self._message = message
        self._kind = kind
        self._data = data
        self._source = source
        self._is_error = is_error

    @staticmethod
    def agent(message: str, data: Any = None, source: str = "", is_error: bool = False) -> ChatMessage:
        """
        Constructs a new ChatMessage instance representing a message from an agent.
        This method serves as a convenience static factory function to create a ChatMessage object with the kind set to ChatMessageKind.Agent, which signifies that the message originates from an agent.
        
        Args:
            message (str):
                 The content of the agent's message.
            data (Any, optional):
                 Additional data associated with the message. Defaults to None.
            source (str, optional):
                 The identifier of the source where the message originated. Defaults to an empty string.
            is_error (bool, optional):
                 A flag indicating whether the message represents an error. Defaults to False.
        
        Returns:
            (ChatMessage):
                 An instance of a ChatMessage encapsulating the agent's message details.

        """
        return ChatMessage(message, ChatMessageKind.Agent, data, source, is_error)

    @staticmethod
    def user(message: str, data: Any = None, source: str = "", is_error: bool = False) -> ChatMessage:
        """
        Creates a `ChatMessage` instance representing a message from a user.
        This static method initializes a new `ChatMessage` object with the `ChatMessageKind.User` kind attribute, indicating that the message originated from a user. It is one of the specialized constructors tailored to create messages of different kinds through static methods.
        
        Args:
            message (str):
                 The content of the user's message.
            data (Any, optional):
                 Additional data associated with the user's message. Defaults to None.
            source (str, optional):
                 The source identifier from where the user's message originated. Defaults to an empty string.
            is_error (bool, optional):
                 Flag indicating if the message represents an error. Defaults to False.
        
        Returns:
            (ChatMessage):
                 An instance of `ChatMessage` with the kind set to `ChatMessageKind.User`.

        """
        return ChatMessage(message, ChatMessageKind.User, data, source, is_error)

    @staticmethod
    def skill(message: str, data: Any = None, source: str = "", is_error: bool = False) -> ChatMessage:
        """
        Creates a `ChatMessage` instance representing a message of 'Skill' kind.
        This method is a convenience static factory creating `ChatMessage` objects that
        are pre-filled with the `ChatMessageKind.Skill` type. Additional message-related
        data can be passed if required.
        
        Args:
            message (str):
                 The text of the chat message.
            data (Any, optional):
                 Additional data associated with the message. Defaults to None.
            source (str, optional):
                 The origin identifier of the message, indicating where it came from. Defaults to an empty string.
            is_error (bool, optional):
                 Flag indicating if the message represents an error state. Defaults to False.
        
        Returns:
            (ChatMessage):
                 A new instance of ChatMessage with the kind set to `ChatMessageKind.Skill`.
            

        """
        return ChatMessage(message, ChatMessageKind.Skill, data, source, is_error)

    @staticmethod
    def chain(message: str, data: Any = None, source: str = "", is_error: bool = False) -> ChatMessage:
        """
        Creates a new ChatMessage instance representing a chain-type message.
        This static method is a factory function for creating ChatMessage objects categorized as 'Chain' type. It wraps the input data and meta-information about the message into a new instance of ChatMessage with its kind attribute set as ChatMessageKind.Chain.
        
        Args:
            message (str):
                 The actual text content of the chat message.
            data (Any, optional):
                 Additional data or context associated with the message. Defaults to None.
            source (str, optional):
                 Identifier for the original source of the message. Defaults to an empty string.
            is_error (bool, optional):
                 Indicates if the message represents an error state. Defaults to False.
        
        Returns:
            (ChatMessage):
                 An instance of ChatMessage with kind attribute set to ChatMessageKind.Chain.
            

        """
        return ChatMessage(message, ChatMessageKind.Chain, data, source, is_error)

    @property
    def message(self) -> str:
        """
        The 'message' property is a getter that retrieves the private '_message' attribute.
        This function is a property getter, which means that it can be accessed like an attribute.
        It is used to safely access the value of the '_message' attribute from outside the class.
        This prevents direct modification of the underlying private variable, which helps to
        maintain the integrity of the encapsulated data. As this function is marked with
        the @property decorator, it can be used without parentheses.
        
        Returns:
            (str):
                 The current value of the '_message' attribute.
            

        """
        return self._message

    @property
    def kind(self) -> ChatMessageKind:
        """
        
        Returns the kind of the chat message.
            This property is used to retrieve the specific type of chat message it represents. The type is encapsulated within the ChatMessageKind enum, defining various possible kinds.
        
        Returns:
            (ChatMessageKind):
                 An enumeration member indicating the specific kind of chat message.
            

        """
        return self._kind

    @property
    def is_kind_skill(self) -> bool:
        """
        
        Returns whether the message kind is 'Skill'.
            This property checks if the current instance's kind attribute is equal to the 'Skill' enum value in ChatMessageKind.
        
        Returns:
            (bool):
                 True if the kind is 'Skill', False otherwise.

        """
        return self._kind == ChatMessageKind.Skill

    @property
    def is_kind_agent(self) -> bool:
        """
        
        Returns whether the message kind is from an agent.
            This read-only property checks the kind of the chat message and determines if it is sent by an agent by comparing the kind attribute with ChatMessageKind.Agent.
        
        Returns:
            (bool):
                 True if the chat message kind is ChatMessageKind.Agent, False otherwise.

        """
        return self._kind == ChatMessageKind.Agent

    @property
    def is_kind_chain(self) -> bool:
        """
        Checks if the chat message kind is 'Chain'.
        This method is a property that when accessed, indicates whether the message kind
        is a 'Chain' type. It is used to identify chained messages within a conversation.
        
        Returns:
            (bool):
                 True if the message kind is 'Chain', otherwise False.

        """
        return self._kind == ChatMessageKind.Chain

    @property
    def is_kind_user(self) -> bool:
        """
        Checks if the message kind is 'User'.
        This property indicates if the message was sent by a user. If the message kind is `ChatMessageKind.User`, it returns True, otherwise False.
        
        Returns:
            (bool):
                 True if the message kind is 'User', False otherwise.

        """
        return self._kind == ChatMessageKind.User

    @property
    def data(self) -> Any:
        """
        
        Returns the '_data' attribute of the instance.
            This is a property method that returns the data stored within the '_data' attribute. As a property, it allows for the attribute to be accessed as though it is a read-only attribute, meaning it cannot be set or modified directly.
        
        Returns:
            (Any):
                 The data contained within the '_data' attribute of the instance.

        """
        return self._data

    @property
    def source(self) -> str:
        """
        
        Returns the source attribute of the instance.
            This method is a property decorator that provides a getter for the
            private '_source' attribute. It allows for retrieval of the '_source' value
            in a controlled way without direct access or modification of the underlying
            private data.
        
        Returns:
            (str):
                 The value of the '_source' attribute representing the source.

        """
        return self._source

    @property
    def is_ok(self) -> bool:
        """
        Checks if the current state is without errors.
        
        Attributes:
            is_ok (bool):
                 A property that, when accessed, indicates whether
                the current state is error-free. It negates the state of the
                internal `_is_error` flag.
        
        Returns:
            (bool):
                 `True` if there are no errors present (i.e., `_is_error` is `False`),
                otherwise `False`.

        """
        return not self._is_error

    @property
    def is_error(self) -> bool:
        """
        Checks if an error flag is set for the object.
        This property method returns the state of the object's internal error flag. It
        enables external entities to query whether the object has encountered an error.
        
        Returns:
            (bool):
                 True if an error is flagged, False otherwise.

        """
        return self._is_error

    def is_of_kind(self, kind: ChatMessageKind) -> bool:
        """
        Checks if the chat message is of a specified kind.
        This method compares the message's kind with the provided kind parameter to determine if they match.
        
        Args:
            kind (ChatMessageKind):
                 The kind of chat message to check against.
        
        Returns:
            (bool):
                 True if the message is of the provided kind, False otherwise.

        """
        return self._kind == kind

    def is_from_source(self, source: str) -> bool:
        """
        Checks if the object's source matches the provided source string.
        
        Args:
            source (str):
                 The source string to compare with the object's source.
        
        Returns:
            (bool):
                 True if the object's source matches the provided string, False otherwise.

        """
        return self._source == source

    def __str__(self):
        """
        
        Returns a string representation of the object.
            This method overrides the default object's `__str__` method to provide a more descriptive string
            representation of the instance by including the kind and the message attributes.
        
        Returns:
            (str):
                 The string representation of the object, formatted as '<kind>: <message>'.

        """
        return f"{self.kind}: {self.message}"

    def to_string(self, max_length: int = 50):
        """
        Converts the message attribute to a string with a maximum length.
        This method will truncate the message to `max_length` and append an ellipsis if it exceeds `max_length`. If it does not exceed `max_length`, it will display the entire message. The string will also include the kind of the message.
        
        Args:
            max_length (int, optional):
                 The maximum number of characters that the message can be before being truncated. Defaults to 50.
        
        Returns:
            (str):
                 A formatted string containing a truncated message (if applicable) and the kind of the message.
            

        """
        message = self.message[:max_length] + "..." if len(self.message) > max_length else self.message
        return f"Message of kind {self.kind}: {message}"

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the instance attributes to a dictionary representation.
        This method takes the error status, kind, message, and source from the instance and constructs a dictionary
        with this data. It's used for serialization or transferring over network calls where an object representation
        is needed in a simple key-value store format.
        
        Returns:
            (Dict[str, Any]):
                 A dictionary with the instance attributes, with keys being the attribute names
                and values being the attribute values.
            

        """
        return {
            "is_error": self.is_error,
            "kind": self.kind.value,
            "message": self.message,
            "source": self.source,
        }


class ScoredChatMessage:
    """
    A class to represent a chat message with an associated score, allowing for comparisons based on the score attribute.
    
    Attributes:
        message (ChatMessage):
             The chat message object.
        score (float):
             A numerical score associated with the chat message, used for ranking or prioritization.
    
    Methods:
        __init__:
             Constructs a ScoredChatMessage instance with a specific message and associated score.
        __gt__:
             Check if this ScoredChatMessage's score is greater than that of another.
        __lt__:
             Check if this ScoredChatMessage's score is less than that of another.
        __ge__:
             Check if this ScoredChatMessage's score is greater than or equal to that of another.
        __le__:
             Check if this ScoredChatMessage's score is less than or equal to that of another.
        __str__:
             Return a string representation of the score for display purposes.
        __repr__:
             Return an 'official' string representation of the ScoredChatMessage object, suitable for debugging.

    """

    message: ChatMessage
    score: float

    def __init__(self, message: ChatMessage, score: float):
        """
        Initializes a new instance of the class with specified message and score.
        
        Args:
            message (ChatMessage):
                 The chat message object that this instance will
                be associated with.
            score (float):
                 The score representing a value associated with the message,
                possibly indicating its relevance or importance.
        
        Attributes:
            message (ChatMessage):
                 The chat message object provided as input.
            score (float):
                 The numeric score associated with the message.
            

        """
        self.message = message
        self.score = score

    def __gt__(self, other: "ScoredChatMessage"):
        """
        Compares the scores of two `ScoredChatMessage` instances to determine if one is greater than the other.
        
        Args:
            other (ScoredChatMessage):
                 Another `ScoredChatMessage` instance to compare against.
        
        Returns:
            (bool):
                 True if the score of the current instance is greater than the score of `other`, otherwise False.

        """
        return self.score > other.score

    def __lt__(self, other: "ScoredChatMessage"):
        """
        Compares this `ScoredChatMessage` object to another `ScoredChatMessage` object to determine
        if the score of the current object is less than the other.
        The `__lt__` method is a special method in Python that corresponds to the '<' operator.
        This method is called to determine the relative ordering between two objects based on
        their respective scores.
        
        Args:
            other (ScoredChatMessage):
                 The `ScoredChatMessage` object to which
                the current object is compared.
        
        Returns:
            (bool):
                 True if the score of the current object is less than the score of 'other',
                False otherwise.
        
        Raises:
            AttributeError:
                 If the 'other' object doesn't have a 'score' attribute.
        
        Note:
            Both the current and the 'other' object must be instances of `ScoredChatMessage`.
            The method assumes that `other` has a 'score' attribute.
            

        """
        return self.score < other.score

    def __ge__(self, other: "ScoredChatMessage"):
        """
        Determines if the score of this `ScoredChatMessage` instance is greater than or equal to the score of
        another `ScoredChatMessage` instance.
        This method overrides the >= (greater than or equal to) operator to compare the scores of two `ScoredChatMessage`
        instances. It is used to implement the comparison logic that allows instances of `ScoredChatMessage` to be
        sorted or checked for equality based on their scores.
        
        Args:
            other (ScoredChatMessage):
                 Another `ScoredChatMessage` instance to compare with.
        
        Returns:
            (bool):
                 True if this instance's score is greater than or equal to `other`'s score, False otherwise.
            

        """
        return self.score >= other.score

    def __le__(self, other: "ScoredChatMessage"):
        """
        Compares this `ScoredChatMessage` object with another to determine if the former's score is less than or equal to the latter's score.
        
        Args:
            other (ScoredChatMessage):
                 The `ScoredChatMessage` object to compare against.
        
        Returns:
            (bool):
                 True if the score of this object is less than or equal to the score of `other`, False otherwise.

        """
        return self.score <= other.score

    def __str__(self):
        """
        Generates a human-readable string representation of the object, indicating its score attribute.
        
        Returns:
            (str):
                 String representation of the object's score.

        """
        return f"{self.score}"

    def __repr__(self):
        """
        Representation method for the class that this function belongs to.
        This magic method overrides the default implementation of '__repr__' to provide
        a more human-readable representation of the object, typically used for debugging.
        It returns a string that looks like a valid Python expression that could be used
        to recreate an object with the same state.
        
        Returns:
            (str):
                 A string that represents the object in a clear format consisting of the
                class name followed by its fields (message and score) in a tuple format.
            

        """
        return f"ScoredChatMessage({self.message}, {self.score})"
