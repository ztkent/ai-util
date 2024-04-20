package aiutil

import (
	"context"
	"fmt"

	"github.com/replicate/replicate-go"
)

type R8Client struct {
	*replicate.Client
	Model       string
	Temperature float32
	Webhook     *replicate.Webhook
}

// Waits for the entire response to be returned
// Adds the users request, and the response to the conversation
func (c *R8Client) SendCompletionRequest(ctx context.Context, conv *Conversation, userPrompt string) (string, error) {
	input := replicate.PredictionInput{
		"prompt":            userPrompt,
		"presence_penalty":  0,
		"frequency_penalty": 0,
		"top_k":             0,
		"top_p":             0.9,
		"temperature":       0.6,
		"length_penalty":    1,
		"max_new_tokens":    512,
		"system_prompt":     "You are a helpful assistant",
		"prompt_template":   "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
	}
	// Run a model and wait for its output
	responseChat, err := c.Run(ctx, c.Model, input, c.Webhook)
	if err != nil {
		return "", err
	} else if responseChat == nil {
		return "", fmt.Errorf("Failed to get response from model")
	}
	fmt.Println(responseChat)
	return responseChat.(string), nil
}

// Available for some models
func (c *R8Client) SendStreamRequest(ctx context.Context, conv *Conversation, userPrompt string, responseChan chan string, errChan chan error) {
	// TODO: Implement this method
	defer close(responseChan)
	defer close(errChan)

	// Ensure we have a conversation to work with
	if conv == nil {
		errChan <- fmt.Errorf("Failed to SendStreamRequest: Conversation is nil")
		return
	}
	res, err := c.SendCompletionRequest(ctx, conv, userPrompt)
	if err != nil {
		errChan <- err
		return
	}
	responseChan <- res
	return
}

func (c *R8Client) SetTemperature(temp float32) {
	if temp >= 0.0 && temp <= 1.0 {
		c.Temperature = temp
	}
}

func (c *R8Client) SetModel(model string) {
	if _, ok := IsReplicateModel(model); ok {
		c.Model = model
	}
}

// Replicate supports a webhook for events, [start, output, logs, completed]
// The client will send an HTTP POST request to "https://example.com/webhook" with information
func (c *R8Client) SetWebhook(url string, events []string) error {
	if len(events) == 0 {
		return fmt.Errorf("No Webhook events provided")
	} else if url == "" {
		return fmt.Errorf("No Webhook URL provided")
	}

	webhookEvents := make([]replicate.WebhookEventType, len(events))
	for i, event := range events {
		webhookEvents[i] = replicate.WebhookEventType(event)
	}
	c.Webhook = &replicate.Webhook{
		URL:    url,
		Events: webhookEvents,
	}
	return nil
}
func (c *R8Client) ListModels(ctx context.Context) ([]string, error) {
	replicateModelPage, err := c.Client.ListModels(ctx)
	if err != nil {
		return nil, err
	}
	models := make([]string, 0)
	for _, model := range replicateModelPage.Results {
		models = append(models, model.Owner+"/"+model.Name+":"+model.LatestVersion.ID)
	}
	return models, nil
}

func (c *R8Client) SetModelWithVersion(ctx context.Context, model string) error {
	replicateModelPage, err := c.Client.ListModels(ctx)
	if err != nil {
		return err
	}
	for _, currModel := range replicateModelPage.Results {
		if currModel.Name == model {
			ver, err := c.GetModelVersion(ctx, currModel.Owner, currModel.Name, currModel.LatestVersion.ID)
			if err != nil {
				return err
			}
			c.Model = currModel.Owner + "/" + currModel.Name + ":" + ver.ID
			return nil
		}
	}

	return fmt.Errorf("Model not found: %s", model)
}
