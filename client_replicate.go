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
		"prompt": userPrompt,
	}

	// Run a model and wait for its output
	responseChat, err := c.Run(ctx, c.Model, input, c.Webhook)
	if err != nil {
		return "", err
	}
	fmt.Println(responseChat)
	return responseChat.(string), nil
}

// Available for some models
func (c *R8Client) SendStreamRequest(ctx context.Context, conv *Conversation, userPrompt string, responseChan chan string, errChan chan error) {
	// TODO: Implement this method
}

func (c *R8Client) SetTemperature(temp float32) {
	// TODO: Implement this method
}

func (c *R8Client) SetModel(model string) {
	// TODO: Implement this method
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
		models = append(models, model.Name)
	}
	return models, nil
}
