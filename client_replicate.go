package aiutil

import (
	"context"
	"fmt"
	"strings"
	"time"

	// Added for stream timeout
	"github.com/replicate/replicate-go"
	"github.com/sashabaranov/go-openai"
)

// R8Client struct wraps the Replicate client and holds its configuration.
type R8Client struct {
	*replicate.Client
	config ClientConfig
}

// GetConfig returns the client's configuration.
func (c *R8Client) GetConfig() ClientConfig {
	return c.config
}

// buildPredictionInput creates the input map from conversation and config.
func (c *R8Client) buildPredictionInput(conv *Conversation, userPrompt string) replicate.PredictionInput {
	input := replicate.PredictionInput{}

	// Map common parameters from ClientConfig
	if c.config.Temperature != nil {
		input["temperature"] = *c.config.Temperature
	}
	if c.config.TopP != nil {
		input["top_p"] = *c.config.TopP
	}
	if c.config.TopK != nil {
		input["top_k"] = *c.config.TopK
	}
	if c.config.Seed != nil {
		input["seed"] = *c.config.Seed
	}
	if c.config.MaxTokens != nil {
		// Replicate often uses 'max_new_tokens' or 'max_length'
		if _, exists := c.config.ReplicateInput["max_new_tokens"]; !exists {
			if _, exists_len := c.config.ReplicateInput["max_length"]; !exists_len {
				input["max_new_tokens"] = *c.config.MaxTokens
			}
		}
	}

	// Merge explicitly provided ReplicateInput, potentially overriding mapped common params
	if c.config.ReplicateInput != nil {
		for k, v := range c.config.ReplicateInput {
			input[k] = v
		}
	}

	// Use the userPrompt directly
	input["prompt"] = userPrompt

	// Extract system prompt if available (first message)
	if len(conv.Messages) > 0 && conv.Messages[0].Role == openai.ChatMessageRoleSystem {
		// Check if 'system_prompt' is already set in ReplicateInput, if so, respect it
		if _, exists := input["system_prompt"]; !exists {
			input["system_prompt"] = conv.Messages[0].Content
		}
	}
	return input
}

// SendCompletionRequest sends a request and waits for the full response.
func (c *R8Client) SendCompletionRequest(ctx context.Context, conv *Conversation, userPrompt string) (string, error) {
	if conv == nil {
		return "", fmt.Errorf("conversation cannot be nil")
	}

	// Add user message
	err := conv.Append(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: userPrompt,
	})
	if err != nil {
		return "", fmt.Errorf("failed to append user prompt: %w", err)
	}

	input := c.buildPredictionInput(conv, userPrompt)

	// Run prediction
	// Use Run() which handles model/version lookup and waits
	output, err := c.Run(ctx, c.config.Model, input, c.config.Webhook)
	if err != nil {
		conv.RemoveLastMessageIfRole(openai.ChatMessageRoleUser)
		return "", fmt.Errorf("replicate run failed for model %s: %w", c.config.Model, err)
	}
	if output == nil {
		conv.RemoveLastMessageIfRole(openai.ChatMessageRoleUser)
		return "", fmt.Errorf("replicate run returned nil output for model %s", c.config.Model)
	}

	// Process output (common pattern: slice of strings)
	var responseStr string
	switch v := output.(type) {
	case string:
		responseStr = v
	case []interface{}:
		var sb strings.Builder
		for _, item := range v {
			if str, ok := item.(string); ok {
				sb.WriteString(str)
			}
		}
		responseStr = sb.String()
	default:
		conv.RemoveLastMessageIfRole(openai.ChatMessageRoleUser)
		return "", fmt.Errorf("replicate run returned unexpected output type (%T) for model %s", output, c.config.Model)
	}

	// Add assistant response
	err = conv.Append(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: responseStr,
	})
	if err != nil {
		conv.RemoveLastMessageIfRole(openai.ChatMessageRoleUser)
		return "", fmt.Errorf("failed to append assistant response: %w", err)
	}

	return responseStr, nil
}

// SendStreamRequest sends a request and streams the response.
func (c *R8Client) SendStreamRequest(ctx context.Context, conv *Conversation, userPrompt string, responseChan chan string, errChan chan error) {
	defer close(responseChan)
	defer close(errChan)
	if conv == nil {
		errChan <- fmt.Errorf("conversation cannot be nil")
		return
	}

	// Add user message
	err := conv.Append(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: userPrompt,
	})
	if err != nil {
		errChan <- fmt.Errorf("failed to append user prompt: %w", err)
		return
	}
	input := c.buildPredictionInput(conv, userPrompt)

	// Extract model owner, name, and version
	modelParts := strings.Split(c.config.Model, ":")
	if len(modelParts) != 2 {
		conv.RemoveLastMessageIfRole(openai.ChatMessageRoleUser)
		errChan <- fmt.Errorf("invalid replicate model format for streaming: %s (expected owner/name:version)", c.config.Model)
		return
	}
	version := modelParts[1]

	// Create prediction with stream enabled
	prediction, err := c.CreatePrediction(ctx, version, input, c.config.Webhook, true)
	if err != nil {
		conv.RemoveLastMessageIfRole(openai.ChatMessageRoleUser)
		errChan <- fmt.Errorf("failed to create replicate prediction: %w", err)
		return
	}

	// Stream the prediction
	var responseBuilder strings.Builder
	streamErrChan := make(chan error, 1) // Buffered channel for stream error
	streamEventChan, streamErrFromChan := c.Client.StreamPrediction(ctx, prediction)

	go func() {
		defer close(streamErrChan)
		// Use the channels returned by StreamPrediction
		for {
			select {
			case event, ok := <-streamEventChan:
				if !ok {
					streamEventChan = nil // Channel closed
				} else {
					switch event.Type {
					case "output":
						responseBuilder.WriteString(event.Data)
						responseChan <- event.Data
					case "error":
						streamErrChan <- fmt.Errorf("replicate stream error: %s", event.Data)
						return // Exit goroutine on stream error event
					case "logs":
						fmt.Println("Stream Log:", event.Data)
					case "done":
					}
				}
			case err, ok := <-streamErrFromChan:
				if !ok {
					streamErrFromChan = nil // Channel closed
				} else if err != nil {
					// Capture error from the stream error channel
					streamErrChan <- fmt.Errorf("replicate stream function failed: %w", err)
					return // Exit goroutine on channel error
				}
			case <-ctx.Done():
				streamErrChan <- fmt.Errorf("context cancelled during stream processing: %w", ctx.Err())
				return // Exit goroutine on context cancellation
			}
			// Exit loop when both channels are closed
			if streamEventChan == nil && streamErrFromChan == nil {
				break
			}
		}
	}()

	// Wait for the stream goroutine to finish or timeout
	select {
	case err := <-streamErrChan:
		if err != nil {
			conv.RemoveLastMessageIfRole(openai.ChatMessageRoleUser)
			errChan <- err // Propagate stream error
			return
		}
	case <-ctx.Done():
		conv.RemoveLastMessageIfRole(openai.ChatMessageRoleUser)
		errChan <- fmt.Errorf("context cancelled during replicate stream: %w", ctx.Err())
		return
	case <-time.After(5 * time.Minute):
		conv.RemoveLastMessageIfRole(openai.ChatMessageRoleUser)
		errChan <- fmt.Errorf("replicate stream timed out")
		return
	}

	// Add the complete assistant response
	responseChat := responseBuilder.String()
	err = conv.Append(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: responseChat,
	})
	if err != nil {
		errChan <- fmt.Errorf("failed to append assistant response post-stream: %w", err)
		return
	}
}

// ListModels lists models available on Replicate.
func (c *R8Client) ListModels(ctx context.Context) ([]string, error) {
	modelPage, err := c.Client.ListModels(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to list replicate models: %w", err)
	}

	models := make([]string, 0, len(modelPage.Results))
	for _, model := range modelPage.Results {
		if model.LatestVersion != nil { // Ensure there's a version
			models = append(models, model.Owner+"/"+model.Name+":"+model.LatestVersion.ID)
		} else {
			models = append(models, model.Owner+"/"+model.Name) // Model without a version?
		}
	}
	return models, nil
}

// SetModelWithVersion resolves the latest version ID for a model like "owner/name"
// and updates the model string in the client's config copy.
func (c *R8Client) SetModelWithVersion(ctx context.Context) error {
	if strings.Contains(c.config.Model, ":") {
		return nil // Version already specified
	}

	modelParts := strings.Split(c.config.Model, "/")
	if len(modelParts) != 2 {
		return fmt.Errorf("invalid replicate model format for version lookup: %s", c.config.Model)
	}
	owner, name := modelParts[0], modelParts[1]

	modelDetails, err := c.Client.GetModel(ctx, owner, name)
	if err != nil {
		return fmt.Errorf("failed to get replicate model details for %s/%s: %w", owner, name, err)
	}

	if modelDetails.LatestVersion == nil || modelDetails.LatestVersion.ID == "" {
		return fmt.Errorf("no latest version found for replicate model %s/%s", owner, name)
	}

	// Update the config's model string
	c.config.Model = modelDetails.Owner + "/" + modelDetails.Name + ":" + modelDetails.LatestVersion.ID
	fmt.Printf("Resolved Replicate model %s/%s to version %s\n", owner, name, modelDetails.LatestVersion.ID) // Info log
	return nil
}
