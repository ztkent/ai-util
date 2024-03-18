package aiclient

import (
	"context"
	"os"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

func MustConnectOpenAI(model OpenAIModel, temperature float32) *Client {
	oaiClient := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
	MustCheckConnection(oaiClient)
	return &Client{Client: oaiClient, Model: model.String(), Temperature: temperature}
}

func MustConnectAnyscale(model AnyscaleModel, temperature float32) *Client {
	config := openai.DefaultConfig(os.Getenv("ANYSCALE_ENDPOINT_TOKEN"))
	config.BaseURL = "https://api.endpoints.anyscale.com/v1"
	asClient := openai.NewClientWithConfig(config)
	MustCheckConnection(asClient)
	return &Client{Client: asClient, Model: model.String(), Temperature: temperature}
}

func MustCheckConnection(client *openai.Client) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := client.ListModels(ctx)
	if err != nil {
		panic(err)
	}
}
