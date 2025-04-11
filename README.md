# AI Util
 Tools for building with AI.

## Features 
- Supported AI Providers:
    - [OpenAI](https://platform.openai.com/docs/overview)
    - [Replicate](https://replicate.com/docs)
- Conversation Controls
- Token Limits
- Resource Management

## Installation
```bash
go get github.com/ztkent/ai-util
```

## Example
```go
    client, _ := aiutil.NewAIClient("openai", "gpt-3.5-turbo", 0.5)
    conversation := aiutil.NewConversation("You are an example assistant.", 100000, true)
    response, _ := client.SendCompletionRequest(CtxWithTimeout, conversation, "Say hello!")
```

## Required API Keys
| Service   | Environment Variable     |
|-----------|--------------------------|
| OpenAI    | `OPENAI_API_KEY`         |
| Replicate | `REPLICATE_API_TOKEN`    |

## Available Models
### OpenAI Models
| Model Name | Model Identifier |
|------------|------------------|
| GPT-3.5 Turbo | `gpt-3.5-turbo` |
| GPT-4 | `gpt-4` |
| GPT-4 Turbo | `gpt-4-turbo` |
| GPT-4o | `gpt-4o` |
| GPT-4o Mini | `gpt-4o-mini` |
| O1 Preview | `o1-preview` |
| O1 Mini | `o1-mini` |

### Replicate Models
| Model Name | Model Identifier |
|------------|------------------|
| Meta Llama 3-8b | `meta/meta-llama-3-8b` |
| Meta Llama 3-70b | `meta/meta-llama-3-70b` |
| Meta Llama 3-8b Instruct | `meta/meta-llama-3-8b-instruct` |
| Meta Llama 3-70b Instruct | `meta/meta-llama-3-70b-instruct` |
| Mistral 7B | `mistralai/mistral-7b-v0.1` |
| Mistral 7B Instruct | `mistralai/mistral-7b-instruct-v0.2` |
| Mixtral 8x7B Instruct | `mistralai/mixtral-8x7b-instruct-v0.1` |
