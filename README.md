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
