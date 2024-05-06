package aiutil

import (
	"context"
	"fmt"

	"github.com/sashabaranov/go-openai"
)

// The OAIClient struct is a wrapper around the OpenAI client
// It can support both OpenAI and Anyscale requests (which use an OpenAI client)
type OAIClient struct {
	*openai.Client
	Model       string
	Temperature float32
}

// Waits for the entire response to be returned
// Adds the users request, and the response to the conversation
func (c *OAIClient) SendCompletionRequest(ctx context.Context, conv *Conversation, userPrompt string) (string, error) {
	// Ensure we have a conversation to work with
	if conv == nil {
		return "", fmt.Errorf("Failed to SendCompletionRequest: Conversation is nil")
	}

	// Add the latest message to the conversation
	err := conv.Append(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: userPrompt,
	})
	if err != nil {
		return "", err
	}

	// Send the request to the LLM 🤖
	completion, err := c.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:       c.Model,
		Messages:    conv.Messages,
		Temperature: c.Temperature,
	})
	if err != nil {
		return "", err
	}
	responseChat := ""
	for _, token := range completion.Choices {
		responseChat = token.Message.Content
	}

	// Add the response to the conversation
	err = conv.Append(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: responseChat,
	})
	if err != nil {
		return "", err
	}
	return responseChat, nil
}

// Stream the response as it comes in
// Adds the users request, and the response to the conversation
func (c *OAIClient) SendStreamRequest(ctx context.Context, conv *Conversation, userPrompt string, responseChan chan string, errChan chan error) {
	defer close(responseChan)
	defer close(errChan)

	// Ensure we have a conversation to work with
	if conv == nil {
		errChan <- fmt.Errorf("Failed to SendStreamRequest: Conversation is nil")
		return
	}

	// Add the latest message to the conversation
	err := conv.Append(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: userPrompt,
	})
	if err != nil {
		errChan <- err
		return
	}

	// Stream the request to the LLM 🤖
	completionStream, err := c.CreateChatCompletionStream(ctx, openai.ChatCompletionRequest{
		Model:       c.Model,
		Temperature: c.Temperature,
		Messages:    conv.Messages,
	})
	if err != nil {
		errChan <- err
		return
	}
	responseChat := ""
	for {
		streamData, err := completionStream.Recv()
		if err != nil {
			break
		}
		for _, token := range streamData.Choices {
			responseChan <- token.Delta.Content
			responseChat += token.Delta.Content
		}
	}

	// Add the response to the conversation, once the stream is closed
	err = conv.Append(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: responseChat,
	})
	if err != nil {
		errChan <- err
		return
	}
	return
}

func (c *OAIClient) GetTemperature() float32 {
	return c.Temperature
}

func (c *OAIClient) SetTemperature(temp float32) {
	if temp >= 0.0 && temp <= 1.0 {
		c.Temperature = temp
	}
}

func (c *OAIClient) GetModel() string {
	return c.Model
}

func (c *OAIClient) SetModel(model string) {
	c.Model = model
}

func (c *OAIClient) SetWebhook(url string, events []string) error {
	return fmt.Errorf("Webhooks are not supported for OpenAI")
}

func (c *OAIClient) ListModels(ctx context.Context) ([]string, error) {
	providerModels, err := c.Client.ListModels(ctx)
	if err != nil {
		return nil, err
	}
	models := make([]string, 0)
	for _, model := range providerModels.Models {
		models = append(models, model.ID)
	}
	return models, nil
}
