# LLMPromptConfigObject

```{eval-rst}
.. autoclass:: council.prompt.LLMPromptConfigObject
```

## Code Example

The following code illustrates the way to load prompt from a YAML file.

```{eval-rst}
.. testcode::

    from council.prompt import LLMPromptConfigObject

    prompt = LLMPromptConfigObject.from_yaml("data/prompts/llm-prompt-sql-template.yaml")
    system_prompt = prompt.get_system_prompt_template("default")
    user_prompt = prompt.get_user_prompt_template("default")
```

# PromptTemplateBase

```{eval-rst}
.. autoclass:: council.prompt.PromptTemplateBase
```

# StringPromptTemplate

```{eval-rst}
.. autoclass:: council.prompt.StringPromptTemplate
```

## Example

```{eval-rst}
.. literalinclude:: ../../../data/prompts/llm-prompt-sql-template.yaml
    :language: yaml
```

# XMLPromptTemplate

```{eval-rst}
.. autoclass:: council.prompt.XMLPromptTemplate
```

## XMLPromptSection

```{eval-rst}
.. autoclass:: council.prompt.XMLPromptSection
```

## Example

```{eval-rst}
.. literalinclude:: ../../../data/prompts/llm-prompt-sql-template-xml.yaml
    :language: yaml
```

With this code:

```{eval-rst}
.. testcode::

    from council.prompt import LLMPromptConfigObject

    prompt = LLMPromptConfigObject.from_yaml("data/prompts/llm-prompt-sql-template-xml.yaml")
    system_prompt_template = prompt.get_system_prompt_template("default")
    print(system_prompt_template)
```

Template will be rendered as follows:

```{eval-rst}
.. testoutput::

    <instructions>
    You are a sql expert solving the `task` 
    leveraging the database schema in the `dataset_description` section.

    - Assess whether the `task` is reasonable and possible
      to solve given the database schema
    - Keep your explanation concise with only important details and assumptions
    </instructions>
    <dataset_description>
    {dataset_description}
    </dataset_description>
    <response_formatting>
    Your entire response must be inside the following code blocks.
    All code blocks are mandatory.
    
    ```solved
    True/False, indicating whether the task is solved
    ```
    
    ```explanation
    String, explanation of the solution if solved or reasoning if not solved
    ```
    
    ```sql
    String, the sql query if the task is solved, otherwise empty
    ```
    </response_formatting>
```
