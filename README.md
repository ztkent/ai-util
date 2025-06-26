# AI Util

Tools for building with AI.

## Features

- Supported AI Providers:
  - [OpenAI](https://platform.openai.com/docs/overview)
  - [Replicate](https://replicate.com/docs)
  - [Google AI](https://ai.google.dev/docs)
- Shared client interface across providers:
  - `Complete` - Single completion requests
  - `Stream` - Streaming completion requests
  - `GetModels` - List available models
- Conversation Management:
  - Manage message history and token counts with auto-truncation
  - Support for system prompts and role-based messaging
- Tool Calling:
  - Invoke backend tools and APIs from within conversations
  - Available only for supported models.

## Installation

```bash
go get github.com/ztkent/ai-util@latest
```

## Configuration Options

**Create a Client:**

```go
client, err := NewAIClient().
    WithOpenAI("api-key").                    // Add OpenAI provider
    WithDefaultProvider("openai").            // Set default provider
    WithDefaultModel("gpt-4o").               // Set default model
    WithDefaultTemperature(0.7).              // Set default temperature
    WithDefaultMaxTokens(4096).               // Set default max tokens
    Build()
```

**Request Options:**

- `Temperature(float64)`: Sampling temperature (0.0 to 2.0)
- `MaxTokens(int)`: Maximum tokens to generate
- `TopP(float64)`: Nucleus sampling probability
- `FrequencyPenalty(float64)`: Penalize frequent tokens (OpenAI)
- `PresencePenalty(float64)`: Penalize present tokens (OpenAI)
- `Stop([]string)`: Stop sequences

**Conversation Options:**

- `SystemPrompt`: Initial system message
- `MaxTokens`: Token limit for conversation
- `AutoTruncate`: Automatically remove old messages when limit reached
- `PreserveSystem`: Keep system message during truncation

## API Keys

API keys are loaded from environment variables by default:

- **OpenAI:** `OPENAI_API_KEY`
- **Google AI:** `GOOGLE_API_KEY` and `GOOGLE_PROJECT_ID`
- **Replicate:** `REPLICATE_API_TOKEN`

Or explicitly provided via the builder pattern.

## Examples

The repository includes examples demonstrating features for each supported provider.

- OpenAI Provider Example (`examples/openai/openai_provider_example.go`)
- Google AI Provider Example (`examples/google/google_provider_example.go`)

- Features:
  - Basic chat completions
  - Streaming responses
  - Tool/function calling
  - Token estimation
  - Model listing
  - Error handling

## Usage

```go
// Basic completion request
resp, err := client.Complete(ctx, &types.CompletionRequest{
    Messages: []*types.Message{
        types.NewTextMessage(types.RoleUser, "What is the capital of France?"),
    },
    Model:       "gpt-4o",
    MaxTokens:   100,
    Temperature: 0.7,
})
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Response: %s\n", resp.Message.TextData)
fmt.Printf("Usage: %+v\n", resp.Usage)
```

```go
// Multi-message conversation
resp, err := client.Complete(ctx, &types.CompletionRequest{
    Messages: []*types.Message{
        types.NewTextMessage(types.RoleSystem, "You are a helpful assistant."),
        types.NewTextMessage(types.RoleUser, "Explain quantum computing in simple terms."),
    },
    Model:       "gpt-4o",
    MaxTokens:   500,
    Temperature: 0.5,
})
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Assistant: %s\n", resp.Message.TextData)
```

### Streaming

```go
// Stream responses
err := client.Stream(ctx, &types.CompletionRequest{
    Messages: []*types.Message{
        types.NewTextMessage(types.RoleUser, "Tell me a story"),
    },
    MaxTokens: 1000,
}, func(ctx context.Context, response *types.StreamResponse) error {
    if response.Delta != nil && response.Delta.TextData != "" {
        fmt.Print(response.Delta.TextData)
    }
    return nil
})
if err != nil {
    log.Fatal(err)
}
```

### Error Handling

The API provides structured error handling:

```go
resp, err := client.Complete(ctx, req)
if err != nil {
    if aiErr, ok := err.(*types.Error); ok {
        fmt.Printf("Provider: %s, Code: %s, Message: %s\n", 
            aiErr.Provider, aiErr.Code, aiErr.Message)
    }
}
```

## Available Models

### OpenAI Models

| Model Name | Model Identifier |
|------------|------------------|
| GPT-4o | `gpt-4o` |
| GPT-4o Mini | `gpt-4o-mini` |
| GPT-4 Turbo | `gpt-4-turbo` |
| GPT-4 | `gpt-4` |
| GPT-3.5 Turbo | `gpt-3.5-turbo` |
| O1 Preview | `o1-preview` |
| O1 Mini | `o1-mini` |

### Google AI Models

| Model Name | Model Identifier | Capabilities |
|------------|------------------|--------------|
| **Gemini 2.5 Series (Latest)** | | |
| Gemini 2.5 Pro | `gemini-2.5-pro` | Chat, Streaming, Tools, Vision, Audio, Video, Thinking |
| Gemini 2.5 Flash | `gemini-2.5-flash` | Chat, Streaming, Tools, Vision, Audio, Video, Thinking |
| Gemini 2.5 Flash-Lite | `gemini-2.5-flash-lite` | Chat, Streaming, Tools, Vision, Audio, Video |
| Gemini 2.5 Flash Preview TTS | `gemini-2.5-flash-preview-tts` | Text-to-Speech |
| Gemini 2.5 Pro Preview TTS | `gemini-2.5-pro-preview-tts` | Text-to-Speech |
| **Gemini 2.0 Series** | | |
| Gemini 2.0 Flash | `gemini-2.0-flash` | Chat, Streaming, Tools, Vision, Audio, Video |
| Gemini 2.0 Flash Preview Image Gen | `gemini-2.0-flash-preview-image-generation` | Chat, Image Generation, Vision, Audio, Video |
| Gemini 2.0 Flash-Lite | `gemini-2.0-flash-lite` | Chat, Streaming, Vision, Audio, Video |
| **Gemini 1.5 Series (Stable)** | | |
| Gemini 1.5 Pro | `gemini-1.5-pro` | Chat, Streaming, Tools, Vision, Audio, Video |
| Gemini 1.5 Flash | `gemini-1.5-flash` | Chat, Streaming, Vision, Audio, Video |
| Gemini 1.5 Flash-8B | `gemini-1.5-flash-8b` | Chat, Streaming, Vision, Audio, Video |
| **Live Interaction Models** | | |
| Gemini 2.5 Flash Live | `gemini-2.5-flash-live` | Live Audio/Video, Streaming |
| Gemini 2.0 Flash Live | `gemini-2.0-flash-live` | Live Audio/Video, Streaming |
| **Embedding Models** | | |
| Text Embedding 004 | `text-embedding-004` | Text Embeddings |
| Gemini Embedding Experimental | `gemini-embedding-exp` | Text Embeddings |
| **Generation Models** | | |
| Imagen 4 | `imagen-4.0-generate-preview` | Image Generation |
| Imagen 3 | `imagen-3.0-generate-002` | Image Generation |
| Veo 2 | `veo-2.0-generate-001` | Video Generation |

### Replicate Models

| Model Name | Model Identifier |
|------------|------------------|
| Meta Llama 3.1 8B Instruct | `meta/meta-llama-3.1-8b-instruct` |
| Meta Llama 3.1 70B Instruct | `meta/meta-llama-3.1-70b-instruct` |
| Meta Llama 3.1 405B Instruct | `meta/meta-llama-3.1-405b-instruct` |
| Meta Llama 3 8B Instruct | `meta/meta-llama-3-8b-instruct` |
| Meta Llama 3 70B Instruct | `meta/meta-llama-3-70b-instruct` |
| Mistral 7B Instruct | `mistralai/mistral-7b-instruct-v0.2` |
| Mixtral 8x7B Instruct | `mistralai/mixtral-8x7b-instruct-v0.1` |
