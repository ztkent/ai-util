package aiutil

import (
	"context"
	"fmt"
	"io" // Added for stream handling
	"strings"

	"github.com/sashabaranov/go-openai"
)

// OAIClient struct wraps the OpenAI client and holds its configuration.
type OAIClient struct {
	*openai.Client
	config ClientConfig // Store the configuration
}

// GetConfig returns the client's configuration.
func (c *OAIClient) GetConfig() ClientConfig {
	return c.config
}

// buildChatCompletionRequest creates the request struct from conversation and config.
func (c *OAIClient) buildChatCompletionRequest(conv *Conversation) openai.ChatCompletionRequest {
	req := openai.ChatCompletionRequest{
		Model:    c.config.Model,
		Messages: conv.Messages, // Use all messages from conversation
	}
	// Apply config options if they are set (non-nil pointers)
	if c.config.Temperature != nil {
		req.Temperature = float32(*c.config.Temperature)
	}
	if c.config.TopP != nil {
		req.TopP = float32(*c.config.TopP)
	}
	if c.config.MaxTokens != nil {
		req.MaxTokens = *c.config.MaxTokens
	}
	if c.config.PresencePenalty != nil {
		req.PresencePenalty = float32(*c.config.PresencePenalty)
	}
	if c.config.FrequencyPenalty != nil {
		req.FrequencyPenalty = float32(*c.config.FrequencyPenalty)
	}
	if c.config.Seed != nil {
		req.Seed = c.config.Seed
	}
	if c.config.ResponseFormat != "" {
		req.ResponseFormat = &openai.ChatCompletionResponseFormat{Type: openai.ChatCompletionResponseFormatType(c.config.ResponseFormat)}
	}
	if c.config.User != "" {
		req.User = c.config.User
	}
	// Add other parameters like Stop, LogitBias etc. if needed in ClientConfig/Options

	return req
}

// SendCompletionRequest sends a request and waits for the full response.
func (c *OAIClient) SendCompletionRequest(ctx context.Context, conv *Conversation, userPrompt string) (string, error) {
	if conv == nil {
		return "", fmt.Errorf("conversation cannot be nil")
	}

	// Add the user's message
	err := conv.Append(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: userPrompt,
	})
	if err != nil {
		return "", fmt.Errorf("failed to append user prompt: %w", err)
	}

	// Build and send the request
	req := c.buildChatCompletionRequest(conv)
	completion, err := c.CreateChatCompletion(ctx, req)
	if err != nil {
		// Attempt to remove the user message if the request failed before getting a response
		conv.RemoveLastMessageIfRole(openai.ChatMessageRoleUser)
		return "", fmt.Errorf("failed to create chat completion: %w", err)
	}

	// Check for content filter or other reasons for empty choices
	if len(completion.Choices) == 0 || completion.Choices[0].Message.Content == "" {
		finishReason := ""
		if len(completion.Choices) > 0 {
			finishReason = string(completion.Choices[0].FinishReason)
		}
		// Append an empty assistant message? Or return specific error?
		// Let's return an error indicating no response content.
		conv.RemoveLastMessageIfRole(openai.ChatMessageRoleUser) // Remove user prompt as no valid response pair
		return "", fmt.Errorf("received empty response from model (finish reason: %s)", finishReason)
	}

	responseChat := completion.Choices[0].Message.Content

	// Add the assistant's response
	err = conv.Append(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: responseChat,
	})
	if err != nil {
		// This should ideally not happen if token counting is correct, but handle defensively.
		// Remove the user message as well, as the pair is incomplete.
		conv.RemoveLastMessageIfRole(openai.ChatMessageRoleUser)
		return "", fmt.Errorf("failed to append assistant response (token limit likely exceeded): %w", err)
	}

	return responseChat, nil
}

// SendStreamRequest sends a request and streams the response.
func (c *OAIClient) SendStreamRequest(ctx context.Context, conv *Conversation, userPrompt string, responseChan chan string, errChan chan error) {
	defer close(responseChan)
	defer close(errChan)

	if conv == nil {
		errChan <- fmt.Errorf("conversation cannot be nil")
		return
	}

	// Add the user's message
	err := conv.Append(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: userPrompt,
	})
	if err != nil {
		errChan <- fmt.Errorf("failed to append user prompt: %w", err)
		return
	}

	// Build and send the stream request
	req := c.buildChatCompletionRequest(conv)
	stream, err := c.CreateChatCompletionStream(ctx, req)
	if err != nil {
		conv.RemoveLastMessageIfRole(openai.ChatMessageRoleUser) // Clean up conversation
		errChan <- fmt.Errorf("failed to create chat completion stream: %w", err)
		return
	}
	defer stream.Close()

	var responseBuilder strings.Builder // Use strings.Builder for efficiency
	var finishReason openai.FinishReason

	for {
		response, err := stream.Recv()
		if err == io.EOF {
			break // Stream finished successfully
		}
		if err != nil {
			// Error during stream, try to clean up conversation
			conv.RemoveLastMessageIfRole(openai.ChatMessageRoleUser)
			errChan <- fmt.Errorf("error receiving stream data: %w", err)
			return
		}

		if len(response.Choices) > 0 {
			delta := response.Choices[0].Delta.Content
			responseBuilder.WriteString(delta)
			responseChan <- delta
			finishReason = response.Choices[0].FinishReason
		}
	}

	responseChat := responseBuilder.String()

	// Check if the stream ended for a reason other than "stop" (e.g., length, content_filter)
	if finishReason != "" && finishReason != openai.FinishReasonStop {
		// Handle non-stop finish reasons, maybe log or include in error?
		// For now, proceed to append what was received.
		fmt.Printf("Warning: OpenAI stream finished with reason: %s\n", finishReason)
	}

	// Add the complete assistant response to the conversation
	err = conv.Append(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: responseChat,
	})
	if err != nil {
		// If appending fails (e.g., token limit with stream), send error.
		// The user message is already added, but the pair is incomplete.
		errChan <- fmt.Errorf("failed to append assistant response post-stream (token limit likely exceeded): %w", err)
		return
	}
}

// ListModels lists available OpenAI models.
func (c *OAIClient) ListModels(ctx context.Context) ([]string, error) {
	providerModels, err := c.Client.ListModels(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to list OpenAI models: %w", err)
	}
	models := make([]string, len(providerModels.Models))
	for i, model := range providerModels.Models {
		models[i] = model.ID
	}
	return models, nil
}
