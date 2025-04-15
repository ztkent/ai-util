package aiutil

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/sashabaranov/go-openai"
)

// OAIClient struct wraps the OpenAI client and holds its configuration.
type OAIClient struct {
	*openai.Client
	config ClientConfig
}

// GetConfig returns the client's configuration.
func (c *OAIClient) GetConfig() ClientConfig {
	return c.config
}

// buildChatCompletionRequest creates the request struct from conversation and config.
func (c *OAIClient) buildChatCompletionRequest(conv *Conversation) openai.ChatCompletionRequest {
	req := openai.ChatCompletionRequest{
		Model:    c.config.Model,
		Messages: conv.Messages,
	}
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

	return req
}

// SendCompletionRequest sends a request and waits for the full response.
func (c *OAIClient) SendCompletionRequest(ctx context.Context, conv *Conversation, userPrompt string) (string, error) {
	if conv == nil {
		return "", fmt.Errorf("conversation cannot be nil")
	}

	err := conv.Append(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: userPrompt,
	})
	if err != nil {
		return "", fmt.Errorf("failed to append user prompt: %w", err)
	}

	req := c.buildChatCompletionRequest(conv)
	completion, err := c.CreateChatCompletion(ctx, req)
	if err != nil {
		conv.RemoveLastMessageIfRole(openai.ChatMessageRoleUser)
		return "", fmt.Errorf("failed to create chat completion: %w", err)
	}

	if len(completion.Choices) == 0 || completion.Choices[0].Message.Content == "" {
		finishReason := ""
		if len(completion.Choices) > 0 {
			finishReason = string(completion.Choices[0].FinishReason)
		}
		conv.RemoveLastMessageIfRole(openai.ChatMessageRoleUser)
		return "", fmt.Errorf("received empty response from model (finish reason: %s)", finishReason)
	}

	responseChat := completion.Choices[0].Message.Content

	err = conv.Append(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: responseChat,
	})
	if err != nil {
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

	err := conv.Append(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: userPrompt,
	})
	if err != nil {
		errChan <- fmt.Errorf("failed to append user prompt: %w", err)
		return
	}

	req := c.buildChatCompletionRequest(conv)
	stream, err := c.CreateChatCompletionStream(ctx, req)
	if err != nil {
		conv.RemoveLastMessageIfRole(openai.ChatMessageRoleUser)
		errChan <- fmt.Errorf("failed to create chat completion stream: %w", err)
		return
	}
	defer stream.Close()

	var responseBuilder strings.Builder
	var finishReason openai.FinishReason

	for {
		response, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
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

	if finishReason != "" && finishReason != openai.FinishReasonStop {
		fmt.Printf("Warning: OpenAI stream finished with reason: %s\n", finishReason)
	}

	err = conv.Append(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: responseChat,
	})
	if err != nil {
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
