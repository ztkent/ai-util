package aiutil

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/joho/godotenv"
	"github.com/replicate/replicate-go"
	openai "github.com/sashabaranov/go-openai"
)

const (
	DefaultProvider       = "openai"
	DefaultOpenAIModel    = "turbo35"
	DefaultAnyscaleModel  = "m8x7b"
	DefaultReplicateModel = "gpt-3.5-turbo"
	DefaultTemp           = 0.2
	DefaultMaxTokens      = 100000
)

func NewAIClient(aiProvider string, model string, temperature float64) (Client, error) {
	// Check if we need to switch the provider
	_, isAnyscaleModel := IsAnyscaleModel(model)
	_, isReplicateModel := IsReplicateModel(model)
	if isAnyscaleModel {
		aiProvider = "anyscale"
	} else if isReplicateModel {
		aiProvider = "replicate"
	}

	var client Client
	if aiProvider == "openai" {
		err := MustLoadAPIKey(OpenAI)
		if err != nil {
			return nil, fmt.Errorf("Failed to load OpenAI API key: %s", err)
		}
		// Set default model if none is provided
		if model == "" {
			model = DefaultOpenAIModel
		}
		if model, ok := IsOpenAIModel(model); ok {
			client = MustConnectOpenAI(model, float32(temperature))
		} else {
			return nil, fmt.Errorf("Invalid OpenAI model: %s provided", model)
		}
	} else if aiProvider == "anyscale" {
		err := MustLoadAPIKey(Anyscale)
		if err != nil {
			return nil, fmt.Errorf("Failed to load Anyscale API key: %s", err)
		}
		// Set default model if none is provided
		if model == "" {
			model = DefaultAnyscaleModel
		}
		if model, ok := IsAnyscaleModel(model); ok {
			client = MustConnectAnyscale(model, float32(temperature))
		} else {
			return nil, fmt.Errorf("Invalid Anyscale model: %s provided", model)
		}
	} else if aiProvider == "replicate" {
		err := MustLoadAPIKey(Replicate)
		if err != nil {
			return nil, fmt.Errorf("Failed to load Replicate API key: %s", err)
		}
		// Set default model if none is provided
		if model == "" {
			model = DefaultReplicateModel
		}
		if model, ok := IsReplicateModel(model); ok {
			client = MustConnectReplicate(model, float32(temperature))
		} else {
			return nil, fmt.Errorf("Invalid Replicate model: %s provided", model)
		}
	} else {
		return nil, fmt.Errorf("Invalid AI provider: %s provided, select either anyscale or openai", aiProvider)
	}
	return client, nil
}

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

func MustConnectReplicate(model ReplicateModel, temperature float32) Client {
	r8, err := replicate.NewClient(replicate.WithTokenFromEnv()) // REPLICATE_API_TOKEN
	if err != nil {
		panic(fmt.Errorf("Failed to create Replicate client: %v", err))
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
