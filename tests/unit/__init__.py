class LLMModels:
    Azure: str = "azure-llmodel.yaml"
    OpenAI: str = "openai-llmodel.yaml"
    Anthropic: str = "anthropic-llmodel.yaml"
    AzureWithFallback: str = "azure-with-fallback-llmodel.yaml"
    Gemini: str = "gemini-llmodel.yaml"
    Ollama: str = "ollama-llmodel.yaml"
    Groq: str = "groq-llmodel.yaml"


class LLMPrompts:
    sample: str = "prompt-abc.yaml"
    sql: str = "prompt-sql.yaml"
    sql_template: str = "prompt-template-sql.yaml"
    large: str = "prompt-large.yaml"


class XMLPrompts:
    sample: str = "xml-prompt-abc.yaml"
    sql_template: str = "xml-prompt-template-sql.yaml"


class LLMDatasets:
    batch: str = "dataset-batch.yaml"
    finetuning: str = "dataset-fine-tuning.yaml"
