# ask-llm

A model-agnostic Python library providing a unified API for interacting with OpenAI and Google Gemini LLMs.

## Features

- **Model-agnostic**: Automatically selects the appropriate provider based on model name
- **Unified API**: Same interface for OpenAI and Google Gemini
- **Tool Calling**: Full support for OpenAI's tool calling with multi-step execution
- **Structured Outputs**: Support for JSON schema and response formatting
- **Caching**: Built-in prompt caching for both providers
- **Citations**: Automatic citation injection for Google Gemini responses
- **Reasoning**: Support for OpenAI's reasoning models with effort control

## Installation

```bash
pip install ask-llm
```

## Quick Start

```python
import asyncio
from ask_llm import AskLLM

async def main():
    # Initialize with any model (model is required)
    llm = AskLLM(model="gpt-4o-mini")
    
    # Simple question
    response = await llm.ask(
        prompt="What is Python?",
        system_instruct="You are a helpful assistant."
    )
    print(response)
    
    # Use Google Gemini
    llm_gemini = AskLLM(model="gemini-2.0-flash-exp")
    response = await llm_gemini.ask(
        prompt="Explain quantum computing",
        system_instruct="You are a physics expert."
    )
    print(response)

asyncio.run(main())
```

## Configuration

Set environment variables for API keys:

```bash
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"
```

## Usage Examples

### Basic Usage

```python
from ask_llm import AskLLM

# Model parameter is required
llm = AskLLM(model="gpt-4o-mini")
response = await llm.ask(prompt="Hello, world!")
```

### With System Instructions

```python
response = await llm.ask(
    prompt="Write a haiku about coding",
    system_instruct="You are a creative poet."
)
```

### Structured Outputs (JSON Schema)

```python
response = await llm.ask(
    prompt="Extract key information from: 'John Doe, age 30, works at Acme Corp'",
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person_info",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number"},
                    "company": {"type": "string"}
                }
            }
        }
    }
)
```

### Tool Calling (OpenAI)

```python
def get_weather(location: str) -> dict:
    # Your tool implementation
    return {"temperature": 72, "condition": "sunny"}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    }
]

async def execute_tool(name: str, args: dict) -> dict:
    if name == "get_weather":
        result = get_weather(args.get("location", ""))
        return {"ok": True, "result": result}
    return {"ok": False, "error": "Unknown tool"}

response = await llm.ask(
    prompt="What's the weather in San Francisco?",
    tools_schema=tools,
    execute_tool_cb=execute_tool
)
```

### Reasoning Models (OpenAI)

```python
llm = AskLLM(model="o1-preview")
response = await llm.ask(
    prompt="Solve this math problem: 2x + 5 = 15",
    reasoning_effort="high"
)
```

## Supported Models

### OpenAI
- `gpt-4o`, `gpt-4o-mini`
- `gpt-4-turbo`, `gpt-4`
- `o1-preview`, `o1-mini`
- Any OpenAI model name

### Anthropic Claude
- `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5` (latest)
- `claude-3-5-sonnet`, `claude-3-opus`, `claude-3-sonnet`
- Any Claude model name (claude-* prefix)

### Google Gemini
- `gemini-2.0-flash-exp`
- `gemini-1.5-pro`, `gemini-1.5-flash`
- Any Google Gemini model name

## API Reference

### `AskLLM`

#### `__init__(model: str)`

Initialize the LLM client.

- `model`: Model name (provider auto-detected, required)

#### `ask(...)`

Generate a response from the LLM.

**Parameters:**
- `prompt` (str): User prompt/question
- `system_instruct` (str, optional): System instruction
- `max_tokens` (int, optional): Maximum tokens to generate
- `temperature` (float, optional): Sampling temperature (0-2)
- `top_p` (float, optional): Nucleus sampling parameter
- `presence_penalty` (float, optional): Presence penalty (OpenAI only)
- `reasoning_effort` (str, optional): Reasoning effort level (OpenAI only)
- `tools_schema` (list, optional): Tool/function calling schema (OpenAI, Anthropic, Google)
- `response_format` (dict, optional): Response format specification
- `execute_tool_cb` (callable, optional): Tool execution callback (OpenAI, Anthropic, Google)
- `max_steps` (int, optional): Maximum tool-calling steps (OpenAI only, default: 24)
- `max_effective_tool_steps` (int, optional): Maximum effective tool steps (OpenAI only, default: 12)

**Returns:** `str` - Generated text response

**Raises:**
- `ValidationError`: If prompt is empty or invalid parameters provided
- `APIError`: If the API call fails
- `ConfigurationError`: If API keys are missing or client initialization fails

## Environment Variables

- `OPENAI_API_KEY`: OpenAI API key (required for OpenAI models)
- `ANTHROPIC_API_KEY`: Anthropic API key (required for Claude models)
- `GOOGLE_API_KEY`: Google API key (required for Google models)
- `GOOGLE_EXPLICIT_CACHE`: Enable Google context caching (default: "1")
- `GOOGLE_INLINE_CITATIONS`: Enable inline citations (default: "1")

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
