package aiutil

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/joho/godotenv"
	"github.com/replicate/replicate-go"
	openai "github.com/sashabaranov/go-openai"
)

const (
	DefaultMaxTokens = 32768
	DefaultTemp      = 0.7
)

// NewAIClient creates a new AI client based on the provided options.
// It requires at least WithProvider() and WithModel() options.
// API keys should be provided via WithAPIKey() or environment variables
// (OPENAI_API_KEY or REPLICATE_API_TOKEN).
func NewAIClient(opts ...Option) (Client, error) {
	config := ClientConfig{
		HTTPClient: http.DefaultClient,
	}

	for _, opt := range opts {
		opt(&config)
	}

	if config.Provider == "" {
		return nil, fmt.Errorf("provider is required (use WithProvider)")
	}

	if config.APIKey == "" {
		var envKey string
		switch Provider(config.Provider) {
		case OpenAI:
			envKey = "OPENAI_API_KEY"
		case Replicate:
			envKey = "REPLICATE_API_TOKEN"
		default:
			return nil, fmt.Errorf("unknown provider: %s", config.Provider)
		}
		if os.Getenv(envKey) == "" {
			_ = godotenv.Load() // Ignore error if .env doesn't exist
		}
		config.APIKey = os.Getenv(envKey)
		if config.APIKey == "" {
			return nil, fmt.Errorf("API key for provider %s not found (set %s env var or use WithAPIKey)", config.Provider, envKey)
		}
	}

	switch Provider(config.Provider) {
	case OpenAI:
		return ConnectOpenAI(&config)
	case Replicate:
		return ConnectReplicate(&config)
	default:
		return nil, fmt.Errorf("unsupported provider: %s", config.Provider)
	}
}

// ConnectOpenAI establishes a connection with the OpenAI API.
func ConnectOpenAI(config *ClientConfig) (Client, error) {
	if config.Model == "" {
		config.Model = GPT41.String()
	} else if _, ok := IsSupportedOpenAIModel(config.Model); !ok {
		return nil, fmt.Errorf("unsupported OpenAI model specified: %s", config.Model)
	}

	oaiConfig := openai.DefaultConfig(config.APIKey)
	if config.BaseURL != "" {
		oaiConfig.BaseURL = config.BaseURL
	}
	if config.OrgID != "" {
		oaiConfig.OrgID = config.OrgID
	}
	if config.HTTPClient != nil {
		oaiConfig.HTTPClient = config.HTTPClient
	}

	oaiClient := openai.NewClientWithConfig(oaiConfig)
	client := &OAIClient{
		Client: oaiClient,
		config: *config,
	}

	return client, CheckConnection(client)
}

// ConnectReplicate establishes a connection with the Replicate API.
func ConnectReplicate(config *ClientConfig) (Client, error) {
	if config.Model == "" {
		config.Model = MetaLlama38bInstruct.String()
	} else if _, ok := IsSupportedReplicateModel(config.Model); !ok {
		if !strings.Contains(config.Model, "/") {
			return nil, fmt.Errorf("invalid Replicate model format: %s (expected 'owner/name' or 'owner/name:version')", config.Model)
		}
	}

	opts := []replicate.ClientOption{replicate.WithToken(config.APIKey)}
	if config.BaseURL != "" {
		opts = append(opts, replicate.WithBaseURL(config.BaseURL))
	}

	r8, err := replicate.NewClient(opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create Replicate client: %w", err)
	}

	client := &R8Client{
		Client: r8,
		config: *config,
	}

	if !strings.Contains(client.config.Model, ":") {
		err = client.SetModelWithVersion(context.Background())
		if err != nil {
			fmt.Printf("Warning: Failed to resolve latest version for Replicate model %s: %v\n", client.config.Model, err)
		}
	}

	return client, nil
}

// CheckConnection attempts a simple API call to verify connectivity and authentication.
func CheckConnection(client Client) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_, err := client.ListModels(ctx)
	if err != nil {
		return fmt.Errorf("connection check failed: %w", err)
	}
	return nil
}
