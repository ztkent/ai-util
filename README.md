# AI Util
Provides a unified platform to interact with a range of AI services.

## Features 
- Services:
    - [OpenAI](https://platform.openai.com/docs/overview)
    - [Replicate](https://replicate.com/docs)
    - [Anyscale](https://docs.endpoints.anyscale.com/)
- Conversation Controls
- Token Management + Limits
- Simple RAG

## Installation
```bash
go get github.com/ztkent/ai-util
```

## Usage
#### Required API Keys
| Service   | Environment Variable     |
|-----------|--------------------------|
| OpenAI    | `OPENAI_API_KEY`         |
| Replicate | `REPLICATE_API_TOKEN`    |
| Anyscale  | `ANYSCALE_ENDPOINT_TOKEN`|

## Available Models
### OpenAI Models
| Model Name | Model Identifier | Cost (IN/OUT per 1M tokens) |
|------------|------------------|-----------------------------|
| GPT-3.5 Turbo | `gpt-3.5-turbo` | - |
| GPT-4 Turbo Preview | `gpt-4-turbo-preview` | - |
| GPT-4 Turbo | `gpt-4-turbo` | - |

### Replicate Models
| Model Name | Model Identifier | Cost (IN/OUT per 1M tokens) |
|------------|------------------|-----------------------------|
| Meta Llama 2-70b | `meta/llama-2-70b` | $0.65 / $2.75 |
| Meta Llama 2-13b | `meta/llama-2-13b` | $0.10 / $0.50 |
| Meta Llama 2-7b | `meta/llama-2-7b` | $0.05 / $0.25 |
| Meta Llama 2-70b Chat | `meta/llama-2-70b-chat` | $0.65 / $2.75 |
| Meta Llama 2-13b Chat | `meta/llama-2-13b-chat` | $0.10 / $0.50 |
| Meta Llama 2-7b Chat | `meta/llama-2-7b-chat` | $0.05 / $0.25 |
| Meta Llama 3-8b | `meta/meta-llama-3-8b` | $0.05 / $0.25 |
| Meta Llama 3-70b | `meta/meta-llama-3-70b` | $0.65 / $2.75 |
| Meta Llama 3-8b Instruct | `meta/meta-llama-3-8b-instruct` | $0.05 / $0.25 |
| Meta Llama 3-70b Instruct | `meta/meta-llama-3-70b-instruct` | $0.65 / $2.75 |
| Mistral 7B | `mistralai/mistral-7b-v0.1` | $0.05 / $0.25 |
| Mistral 7B Instruct | `mistralai/mistral-7b-instruct-v0.2` | $0.05 / $0.25 |
| Mixtral 8x7B Instruct | `mistralai/mixtral-8x7b-instruct-v0.1` | $0.30 / $1.00 |

### Anyscale Models
| Model Name | Model Identifier | Cost (IN/OUT per 1M tokens) |
|------------|------------------|-----------------------------|
| Meta Llama 2-7b Chat | `meta-llama/Llama-2-7b-chat-hf` | - |
| Meta Llama 2-13b Chat | `meta-llama/Llama-2-13b-chat-hf` | - |
| Meta Llama 2-70b Chat | `meta-llama/Llama-2-70b-chat-hf` | - |
| Mistral 7B Instruct | `mistralai/Mistral-7B-Instruct-v0.1` | - |
| Mixtral 8x7B Instruct | `mistralai/Mixtral-8x7B-Instruct-v0.1` | - |
| Code Llama 34b | `codellama/CodeLlama-34b-Instruct-hf` | - |
| Code Llama 70b | `codellama/CodeLlama-70b-Instruct-hf` | - |

