package aiutil

import (
	"context"
	"fmt"
	"strings"

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

// SendStreamRequest sends a streaming request to the model
// https://replicate.com/docs/streaming
// You create a prediction with the stream option.
// Replicate returns a prediction with a URL to receive streaming output.
// You connect to the URL and receive a stream of updates.

func (c *R8Client) SendStreamRequest(ctx context.Context, conv *Conversation, userPrompt string, responseChan chan string, errChan chan error) {
	defer close(responseChan)
	defer close(errChan)

	// Ensure we have a conversation to work with
	if conv == nil {
		errChan <- fmt.Errorf("Failed to SendStreamRequest: Conversation is nil")
		return
	}

	// Create a prediction with the stream option
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
		"stream":            true, // Request streaming output
	}

	// Run a model and wait for its output
	version := strings.Split(c.Model, ":")[1]
	pred, err := c.CreatePrediction(ctx, version, input, c.Webhook, true)
	if err != nil {
		errChan <- err
		return
	}
	// Wait for the prediction to finish
	err = c.Client.Wait(ctx, pred)
	if err != nil {
		errChan <- err
		return
	}
	responseChan <- pred.Output.(string)
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

func (c *R8Client) SetModelWithVersion(ctx context.Context) error {
	modelArgs := strings.Split(c.Model, "/")
	if len(modelArgs) != 2 {
		return fmt.Errorf("Invalid model format: %s", c.Model)
	}

	owner, name := strings.Split(c.Model, "/")[0], strings.Split(c.Model, "/")[1]
	currModel, err := c.Client.GetModel(ctx, owner, name)
	if err != nil {
		return err
	}
	c.Model = currModel.Owner + "/" + currModel.Name + ":" + currModel.LatestVersion.ID
	return nil
}
