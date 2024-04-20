package aiutil

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/joho/godotenv"
	"github.com/replicate/replicate-go"
	openai "github.com/sashabaranov/go-openai"
)

func MustConnectOpenAI(model OpenAIModel, temperature float32) Client {
	oaiutil := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
	client := &OAIClient{Client: oaiutil, Model: model.String(), Temperature: temperature}
	MustCheckConnection(client)
	return client
}

func MustConnectAnyscale(model AnyscaleModel, temperature float32) Client {
	config := openai.DefaultConfig(os.Getenv("ANYSCALE_ENDPOINT_TOKEN"))
	config.BaseURL = "https://api.endpoints.anyscale.com/v1"
	asClient := openai.NewClientWithConfig(config)
	client := &OAIClient{Client: asClient, Model: model.String(), Temperature: temperature}
	MustCheckConnection(client)
	return client
}

func MustConnectReplicate(model AnyscaleModel, temperature float32) Client {
	r8, err := replicate.NewClient(replicate.WithTokenFromEnv()) // REPLICATE_API_TOKEN
	if err != nil {
		log.Fatalf("Failed to create Replicate client: %v", err)
	}
	client := &R8Client{Client: r8, Model: model.String(), Temperature: temperature}
	MustCheckConnection(client)
	return client
}

func MustCheckConnection(client Client) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := client.ListModels(ctx)
	if err != nil {
		panic(err)
	}
}

// Ensure we have the right env variables set for the given source
func MustLoadAPIKey(provider Provider) error {
	// Load the .env file if we don't have the env var set
	loadEnvVar := func(varName string) error {
		if os.Getenv(varName) == "" {
			err := godotenv.Load()
			if err != nil || os.Getenv(varName) == "" {
				return fmt.Errorf("Failed to load %s", varName)
			}
		}
		return nil
	}
	switch provider {
	case OpenAI:
		{
			if err := loadEnvVar("OPENAI_API_KEY"); err != nil {
				return err
			}
		}
	case Anyscale:
		{
			if err := loadEnvVar("ANYSCALE_ENDPOINT_TOKEN"); err != nil {
				return err
			}
		}
	case Replicate:
		{
			if err := loadEnvVar("REPLICATE_API_TOKEN"); err != nil {
				return err
			}
		}
	}

	return nil
}
