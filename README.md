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

#### Pricing (04/12/2024)
```
gpt-3.5-turbo-0125			Input: $0.50 / 1M tokens 	Output: $1.50 / 1M tokens
mixtral-8x7B-instruct   	Input: $0.50 / 1M tokens 	Output: $0.50 / 1M tokens
gpt-4-turbo-2024-04-09		Input: $10.00 / 1M tokens	Output: $30.00 / 1M tokens
```