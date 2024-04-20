package aiutil

import (
	"context"
)

type Client interface {
	SendCompletionRequest(ctx context.Context, conv *Conversation, userPrompt string) (string, error)
	SendStreamRequest(ctx context.Context, conv *Conversation, userPrompt string, responseChan chan string, errChan chan error)
	SetTemperature(temp float32)
	SetModel(model string)
	SetWebhook(url string, events []string) error
	ListModels(ctx context.Context) ([]string, error)
}

type Provider string

const (
	OpenAI    Provider = "openai"
	Anyscale  Provider = "anyscale"
	Replicate Provider = "replicate"
)
