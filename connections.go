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
	DefaultOpenAIModel    = GPT4OMini
	DefaultReplicateModel = MetaLlama38b
	DefaultTemp           = 0.2
	DefaultMaxTokens      = 100000
)

func NewAIClient(aiProvider string, model string, temperature float64) (Client, error) {
	defaultModels := map[string]string{
		"openai":    DefaultOpenAIModel.String(),
		"replicate": DefaultReplicateModel.String(),
	}
	connectFuncs := map[string]func(string, float32) (Client, error){
		"openai":    ConnectOpenAI,
		"replicate": ConnectReplicate,
	}
	connectFunc, ok := connectFuncs[aiProvider]
	if !ok {
		return nil, fmt.Errorf("Invalid AI provider: %s provided, select either openai, or replicate", aiProvider)
	}

	// Load the API key for the provider
	err := LoadAPIKey(Provider(aiProvider))
	if err != nil {
		return nil, fmt.Errorf("Failed to load %s API key: %s", aiProvider, err)
	}

	// Use the default model if none is provided
	if model == "" {
		model = defaultModels[aiProvider]
	} else if aiProvider == "openai" {
		if oaiModel, ok := IsSupportedOpenAIModel(model); ok {
			model = oaiModel.String()
		} else {
			return nil, fmt.Errorf("Invalid model: %s provided, select a valid model from the provider: %s", model, aiProvider)
		}
	} else if aiProvider == "replicate" {
		if repModel, ok := IsSupportedReplicateModel(model); ok {
			model = repModel.String()
		} else {
			return nil, fmt.Errorf("Invalid model: %s provided, select a valid model from the provider: %s", model, aiProvider)
		}
	} else {
		return nil, fmt.Errorf("Invalid model: %s provided, select a valid model from the provider: %s", model, aiProvider)
	}

	// Connect to the AI provider
	return connectFunc(model, float32(temperature))
}

func ConnectOpenAI(model string, temperature float32) (Client, error) {
	oaiutil := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
	client := &OAIClient{Client: oaiutil, Model: model, Temperature: temperature}
	return client, CheckConnection(client)
}

func ConnectReplicate(model string, temperature float32) (Client, error) {
	r8, err := replicate.NewClient(replicate.WithTokenFromEnv())
	if err != nil {
		return nil, fmt.Errorf("Failed to create Replicate client: %v", err)
	}
	client := &R8Client{Client: r8, Model: model, Temperature: temperature}
	err = client.SetModelWithVersion(context.Background())
	if err != nil {
		return nil, fmt.Errorf("Failed to set model: %v", err)
	}
	return client, nil
}

func CheckConnection(client Client) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := client.ListModels(ctx)
	if err != nil {
		return err
	}
	return nil
}

// Ensure we have the right env variables set for the given source
func LoadAPIKey(provider Provider) error {
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
	case Replicate:
		{
			if err := loadEnvVar("REPLICATE_API_TOKEN"); err != nil {
				return err
			}
		}
	}

	return nil
}
