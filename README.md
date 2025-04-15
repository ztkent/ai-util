# AI Util

Tools for building with AI.

## Features

- Supported AI Providers:
  - [OpenAI](https://platform.openai.com/docs/overview)
  - [Replicate](https://replicate.com/docs)
- Consistent client interface across providers:
  - `SendCompletionRequest`
  - `SendStreamRequest`
  - `ListModels`
- Conversation Management:
  - Manage message history and token counts.
- Resource Injection:
  - Add content from files or URLs into the conversation context.

## Installation

```bash
go get github.com/ztkent/ai-util@latest
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"
    "time"

    aiutil "github.com/ztkent/ai-util"
)

func main() {
    ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
    defer cancel()

	oaiClient, err := aiutil.NewAIClient(
		aiutil.WithProvider(string(aiutil.OpenAI)),
		aiutil.WithModel(aiutil.GPT41.String()),
		aiutil.WithTemperature(0.7),
	)

	// r8Client, err := aiutil.NewAIClient(
	// 	aiutil.WithProvider(string(aiutil.Replicate)),
	// 	aiutil.WithModel(aiutil.MetaLlama38bInstruct.String()),
	// 	aiutil.WithTemperature(0.6),
	// 	aiutil.WithTopP(0.9),
	// 	aiutil.WithMaxTokens(512),
	// 	aiutil.WithReplicateInput(map[string]interface{}{"presence_penalty": 0.2}),
	// )

	if err != nil {
		log.Fatalf("Failed to create OpenAI client: %v", err)
	}
	conv := aiutil.NewConversation("You are a helpful assistant.", 10000, true)
	
	// Add a file resource (optional)
	err = aiutil.AddFileReference(conv, "./my_document.txt")
	if err != nil {
	    log.Printf("Warning: Failed to add file resource: %v", err)
	}

	// Send a completion request
	prompt := "Explain the concept of recursion in simple terms."
	fmt.Printf("User: %s\n", prompt)
	response, err := oaiClient.SendCompletionRequest(ctx, conv, prompt)
	if err != nil {
		log.Fatalf("OpenAI completion failed: %v", err)
	}
	fmt.Printf("Assistant: %s\n", response)
}
```

## Configuration Options

**Required:**

*   `WithProvider(string)`: Specify the provider (`"openai"` or `"replicate"`). Use `aiutil.OpenAI` or `aiutil.Replicate` constants.
*   `WithModel(string)`: Specify the model identifier (e.g., `aiutil.GPT41.String()`, `"meta/meta-llama-3-8b-instruct"`).

**Common Options:**

*   `WithAPIKey(string)`: Explicitly set the API key (overrides environment variables).
*   `WithBaseURL(string)`: Set a custom base URL (e.g., for proxies or self-hosted models).
*   `WithTemperature(float64)`: Sampling temperature (e.g., 0.7).
*   `WithTopP(float64)`: Nucleus sampling probability (e.g., 0.9).
*   `WithSeed(int)`: Seed for potentially deterministic outputs (if supported by the model).
*   `WithMaxTokens(int)`: Maximum number of tokens to generate in the response.
*   `WithHTTPClient(*http.Client)`: Provide a custom HTTP client.

**OpenAI Specific Options:**

*   `WithOpenAIOrganizationID(string)`
*   `WithOpenAIPresencePenalty(float64)`
*   `WithOpenAIFrequencyPenalty(float64)`
*   `WithOpenAIResponseFormat(string)`: E.g., `"json_object"`.
*   `WithOpenAIUser(string)`: Unique identifier for the end-user.

**Replicate Specific Options:**

*   `WithReplicateTopK(int)`
*   `WithReplicateInput(map[string]interface{})`: Set arbitrary model inputs. Merges with other inputs.
*   `WithReplicateWebhook(string, []replicate.WebhookEventType)`: Configure webhooks for prediction events.

## API Keys

API keys are typically loaded from environment variables:

*   **OpenAI:** `OPENAI_API_KEY`
*   **Replicate:** `REPLICATE_API_TOKEN`

## Resource Management

Utility functions allow adding external content as system messages:

* `AddFileReference(conv *Conversation, path string)`: Reads content from a local file.
* `AddURLReference(conv *Conversation, urlStr string)`: Fetches and extracts text content from a URL.

Content is truncated if it exceeds `MaxResourceContentLength`.  
Resource management must be enabled in the `Conversation`.

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
| GPT-4.1 | `gpt-4.1` |

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