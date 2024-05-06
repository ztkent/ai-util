package aiutil

import (
	"context"
	"fmt"
	"strings"

	"github.com/replicate/replicate-go"
	"github.com/sashabaranov/go-openai"
)

// TODO: Fix prompt and conversation

type R8Client struct {
	*replicate.Client
	Model       string
	Temperature float32
	Webhook     *replicate.Webhook
}

// Waits for the entire response to be returned
// Adds the users request, and the response to the conversation
func (c *R8Client) SendCompletionRequest(ctx context.Context, conv *Conversation, userPrompt string) (string, error) {
	// Add the latest message to the conversation
	err := conv.Append(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: userPrompt,
	})
	if err != nil {
		return "", err
	}

	input := replicate.PredictionInput{
		"prompt":            conv.History() + "Current Request: " + userPrompt,
		"presence_penalty":  0,
		"frequency_penalty": 0,
		"top_k":             0,
		"top_p":             0.9,
		"temperature":       0.5,
		"length_penalty":    1,
		"max_new_tokens":    512,
		"system_prompt":     conv.Messages[0].Content,
		"prompt_template": `<|begin_of_text|><|start_header_id|>system<|end_header_id|>
		Answer only the 'Current Request'.
		<|eot_id|><|start_header_id|>user<|end_header_id|>		
		{prompt} 
		<|eot_id|><|start_header_id|><|end_header_id|>`,
	}

	// Run a model and wait for its output
	responseChat, err := c.Run(ctx, c.Model, input, c.Webhook)
	if err != nil {
		return "", err
	} else if responseChat == nil {
		return "", fmt.Errorf("Failed to get response from model")
	}
	// Check and convert responseChat to a string
	var responseStr string
	switch v := responseChat.(type) {
	case string:
		responseStr = v
	case []interface{}:
		var strSlice []string
		for _, item := range v {
			str, ok := item.(string)
			if !ok {
				return "", fmt.Errorf("item in responseChat slice is not a string")
			}
			strSlice = append(strSlice, str)
		}
		responseStr = strings.Join(strSlice, "")
	default:
		return "", fmt.Errorf("responseChat is invalid type")
	}

	// Add the response to the conversation
	err = conv.Append(openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: responseStr,
	})
	if err != nil {
		return "", err
	}
	return responseStr, nil
}

// SendStreamRequest sends a streaming request to the model
func (c *R8Client) SendStreamRequest(ctx context.Context, conv *Conversation, userPrompt string, responseChan chan string, errChan chan error) {
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

	// Create a prediction with the stream option
	input := replicate.PredictionInput{
		"prompt":            conv.History() + "Current Request: " + userPrompt,
		"presence_penalty":  0.4,
		"frequency_penalty": 0,
		"top_k":             0,
		"top_p":             0.9,
		"temperature":       0.5,
		"length_penalty":    1,
		"max_new_tokens":    1024,
		"system_prompt":     conv.Messages[0].Content,
		"prompt_template": `<|begin_of_text|><|start_header_id|>system<|end_header_id|>
		Answer only the 'Current Request'.
		<|eot_id|><|start_header_id|>user<|end_header_id|>		
		{prompt} 
		<|eot_id|><|start_header_id|><|end_header_id|>
		[END]`,
	}

	// Run a model and wait for its output
	version := strings.Split(c.Model, ":")[1]
	pred, err := c.CreatePrediction(ctx, version, input, c.Webhook, true)
	if err != nil {
		errChan <- err
		return
	}

	// Not every model supports streaming, so we should check if it was used
	streamingUsed := false
	responseChat := ""
	streamResChan, streamErrChan := c.Client.StreamPrediction(ctx, pred)
	go func() {
		for {
			select {
			case event := <-streamResChan:
				if event.Type == "output" {
					streamingUsed = true
					// TODO: This should work with [END], but it doesn't..
					if strings.Contains(event.Data, "assistant\n\n") {
						// remove the assistant prompt
						lastToken := event.Data[:strings.LastIndex(event.Data, "assistant")]
						responseChat += lastToken
						responseChan <- lastToken
						return
					}
					responseChat += event.Data
					responseChan <- event.Data
				} else if event.Type == "error" {
					fmt.Println(event.Data)
					errChan <- fmt.Errorf("Error in stream: %s", event.Data)
					return
				} else if event.Type == "done" {
					return
				}

			case err := <-streamErrChan:
				errChan <- err
				return
			}
		}
	}()

	// Wait for the prediction to finish
	err = c.Client.Wait(ctx, pred)
	if err != nil {
		errChan <- err
		return
	}
	if !streamingUsed {
		responseChan <- pred.Output.(string)
		responseChat += pred.Output.(string)
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
}

func (c *R8Client) GetTemperature() float32 {
	return c.Temperature
}

func (c *R8Client) SetTemperature(temp float32) {
	if temp >= 0.0 && temp <= 1.0 {
		c.Temperature = temp
	}
}

func (c *R8Client) GetModel() string {
	return c.Model
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
