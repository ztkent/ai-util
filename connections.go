package aiutil

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/joho/godotenv" // Keep for potential implicit loading if desired, but explicit keys preferred
	"github.com/replicate/replicate-go"
	openai "github.com/sashabaranov/go-openai"
)

const (
	DefaultMaxTokens = 100000 // Default for conversation, not client response
)

// NewAIClient creates a new AI client based on the provided options.
// It requires at least WithProvider() and WithModel() options.
// API keys should be provided via WithAPIKey() or environment variables
// (OPENAI_API_KEY or REPLICATE_API_TOKEN).
func NewAIClient(opts ...Option) (Client, error) {
	// Initialize default config
	config := ClientConfig{
		// Set any sensible defaults here if needed, e.g., HTTPClient
		HTTPClient: http.DefaultClient,
	}

	// Apply options
	for _, opt := range opts {
		opt(&config)
	}

	// Validate required fields
	if config.Provider == "" {
		return nil, fmt.Errorf("provider is required (use WithProvider)")
	}
	// Model validation happens within provider connection logic

	// Load API key from environment if not provided explicitly
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
		// Attempt to load from .env if not set in environment
		if os.Getenv(envKey) == "" {
			_ = godotenv.Load() // Ignore error if .env doesn't exist
		}
		config.APIKey = os.Getenv(envKey)
		if config.APIKey == "" {
			return nil, fmt.Errorf("API key for provider %s not found (set %s env var or use WithAPIKey)", config.Provider, envKey)
		}
	}

	// Select and connect to the provider
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
		config.Model = GPT4OMini.String() // Default OpenAI model if not set
	} else if _, ok := IsSupportedOpenAIModel(config.Model); !ok {
		// Allow potentially unsupported models but maybe warn? Or error? Let's error for now.
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
		config: *config, // Store a copy of the config
	}

	return client, CheckConnection(client)
}

// ConnectReplicate establishes a connection with the Replicate API.
func ConnectReplicate(config *ClientConfig) (Client, error) {
	if config.Model == "" {
		config.Model = MetaLlama38bInstruct.String() // Default Replicate model if not set
	} else if _, ok := IsSupportedReplicateModel(config.Model); !ok {
		// Replicate uses model strings like "owner/name:version" or "owner/name".
		// We don't strictly validate against our enum here, but we need the format.
		if !strings.Contains(config.Model, "/") {
			return nil, fmt.Errorf("invalid Replicate model format: %s (expected 'owner/name' or 'owner/name:version')", config.Model)
		}
	}

	opts := []replicate.ClientOption{replicate.WithToken(config.APIKey)}
	if config.BaseURL != "" {
		opts = append(opts, replicate.WithBaseURL(config.BaseURL))
	}
	if config.HTTPClient != nil {
		// replicate-go doesn't directly support setting http.Client via options easily after v0.10.0
		// Users needing custom clients might need to fork or use lower-level interactions.
		// We'll ignore config.HTTPClient for Replicate for now.
		// Consider logging a warning if config.HTTPClient is set for Replicate.
	}

	r8, err := replicate.NewClient(opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create Replicate client: %w", err)
	}

	client := &R8Client{
		Client: r8,
		config: *config, // Store a copy of the config
	}

	// Resolve model version if not provided
	if !strings.Contains(client.config.Model, ":") {
		err = client.SetModelWithVersion(context.Background())
		if err != nil {
			// Don't fail connection if version resolution fails, maybe model exists without explicit version?
			// Log a warning? For now, proceed.
			fmt.Printf("Warning: Failed to resolve latest version for Replicate model %s: %v\n", client.config.Model, err)
		}
	}

	// No simple connection check like ListModels for Replicate auth validation easily available.
	// Assume connection is okay if client created.
	return client, nil
}

// CheckConnection attempts a simple API call to verify connectivity and authentication.
func CheckConnection(client Client) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second) // Increased timeout
	defer cancel()
	_, err := client.ListModels(ctx) // ListModels works for OpenAI, might need adjustment if other providers added
	if err != nil {
		return fmt.Errorf("connection check failed: %w", err)
	}
	return nil
}

// LoadAPIKey is deprecated, API keys are handled by NewAIClient.
// func LoadAPIKey(provider Provider) error { ... }
