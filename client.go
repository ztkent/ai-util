package aiutil

import (
	"context"
)

type Client interface {
	SendCompletionRequest(ctx context.Context, conv *Conversation, userPrompt string) (string, error)
	SendStreamRequest(ctx context.Context, conv *Conversation, userPrompt string, responseChan chan string, errChan chan error)
	ListModels(ctx context.Context) ([]string, error)
	GetConfig() ClientConfig // Added to access current config

	// Removed methods now handled by options or config
	// GetTemperature() float32
	// SetTemperature(temp float32)
	// GetModel() string
	// SetModel(model string)
	// SetWebhook(url string, events []string) error
}

type Provider string

const (
	OpenAI    Provider = "openai"
	Replicate Provider = "replicate"
)
