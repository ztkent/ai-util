package aiutil

import (
	"fmt"

	"github.com/ztkent/ai-util/providers/google"
	"github.com/ztkent/ai-util/providers/openai"
	"github.com/ztkent/ai-util/providers/replicate"
	"github.com/ztkent/ai-util/types"
)

// AIClient provides a fluent interface for configuring and creating a Client
type AIClient struct {
	config          *ClientConfig
	providerConfigs map[string]types.Config
	middleware      []Middleware
}

// NewAIClient creates a new client builder
func NewAIClient() *AIClient {
	return &AIClient{
		config: &ClientConfig{
			DefaultMaxTokens:   4096,
			DefaultTemperature: 0.7,
			ProviderConfigs:    make(map[string]types.Config),
		},
		providerConfigs: make(map[string]types.Config),
		middleware:      make([]Middleware, 0),
	}
}

// WithDefaultProvider sets the default provider
func (b *AIClient) WithDefaultProvider(provider string) *AIClient {
	b.config.DefaultProvider = provider
	return b
}

// WithDefaultModel sets the default model
func (b *AIClient) WithDefaultModel(model string) *AIClient {
	b.config.DefaultModel = model
	return b
}

// WithDefaultMaxTokens sets the default max tokens
func (b *AIClient) WithDefaultMaxTokens(maxTokens int) *AIClient {
	b.config.DefaultMaxTokens = maxTokens
	return b
}

// WithDefaultTemperature sets the default temperature
func (b *AIClient) WithDefaultTemperature(temperature float64) *AIClient {
	b.config.DefaultTemperature = temperature
	return b
}

// WithOpenAI configures OpenAI provider
func (b *AIClient) WithOpenAI(apiKey string, options ...OpenAIOption) *AIClient {
	config := &openai.Config{
		BaseConfig: types.BaseConfig{
			Provider: "openai",
			APIKey:   apiKey,
		},
	}

	// Apply options
	for _, option := range options {
		option(config)
	}

	b.providerConfigs["openai"] = config
	return b
}

// WithReplicate configures Replicate provider
func (b *AIClient) WithReplicate(apiKey string, options ...ReplicateOption) *AIClient {
	config := &replicate.Config{
		BaseConfig: types.BaseConfig{
			Provider: "replicate",
			APIKey:   apiKey,
		},
		ExtraInputs: make(map[string]interface{}),
	}

	// Apply options
	for _, option := range options {
		option(config)
	}

	b.providerConfigs["replicate"] = config
	return b
}

// WithGoogle configures Google AI provider
func (b *AIClient) WithGoogle(apiKey, projectID string, options ...GoogleOption) *AIClient {
	config := &google.Config{
		BaseConfig: types.BaseConfig{
			Provider: "google",
			APIKey:   apiKey,
		},
		ProjectID: projectID,
		Location:  "us-central1", // Default location
	}

	// Apply options
	for _, option := range options {
		option(config)
	}

	b.providerConfigs["google"] = config
	return b
}

// WithMiddleware adds middleware to the client
func (b *AIClient) WithMiddleware(middleware ...Middleware) *AIClient {
	b.middleware = append(b.middleware, middleware...)
	return b
}

// Build creates and configures the client
func (b *AIClient) Build() (*Client, error) {
	// Set provider configs
	b.config.ProviderConfigs = b.providerConfigs
	b.config.Middleware = b.middleware

	// Create client
	client := NewClient(b.config)

	// Register configured providers
	for providerName := range b.providerConfigs {
		var provider types.Provider

		switch providerName {
		case "openai":
			provider = openai.NewProvider()
		case "replicate":
			provider = replicate.NewProvider()
		case "google":
			provider = google.NewProvider()
		default:
			return nil, fmt.Errorf("unknown provider: %s", providerName)
		}

		if err := client.RegisterProvider(provider); err != nil {
			return nil, fmt.Errorf("failed to register %s provider: %w", providerName, err)
		}
	}

	return client, nil
}

// Option types for provider-specific configuration

// OpenAIOption configures OpenAI-specific settings
type OpenAIOption func(*openai.Config)

// WithOpenAIOrg sets the OpenAI organization ID
func WithOpenAIOrg(orgID string) OpenAIOption {
	return func(c *openai.Config) {
		c.OrgID = orgID
	}
}

// WithOpenAIProject sets the OpenAI project
func WithOpenAIProject(project string) OpenAIOption {
	return func(c *openai.Config) {
		c.Project = project
	}
}

// WithOpenAIBaseURL sets a custom base URL for OpenAI
func WithOpenAIBaseURL(baseURL string) OpenAIOption {
	return func(c *openai.Config) {
		c.BaseURL = baseURL
	}
}

// WithOpenAIUser sets the user identifier for OpenAI
func WithOpenAIUser(user string) OpenAIOption {
	return func(c *openai.Config) {
		c.User = user
	}
}

// ReplicateOption configures Replicate-specific settings
type ReplicateOption func(*replicate.Config)

// WithReplicateWebhook sets the webhook URL for Replicate
func WithReplicateWebhook(webhookURL string) ReplicateOption {
	return func(c *replicate.Config) {
		c.WebhookURL = webhookURL
	}
}

// WithReplicateBaseURL sets a custom base URL for Replicate
func WithReplicateBaseURL(baseURL string) ReplicateOption {
	return func(c *replicate.Config) {
		c.BaseURL = baseURL
	}
}

// WithReplicateExtraInput adds extra input parameters for Replicate
func WithReplicateExtraInput(key string, value interface{}) ReplicateOption {
	return func(c *replicate.Config) {
		if c.ExtraInputs == nil {
			c.ExtraInputs = make(map[string]interface{})
		}
		c.ExtraInputs[key] = value
	}
}

// GoogleOption configures Google AI-specific settings
type GoogleOption func(*google.Config)

// WithGoogleLocation sets the Google Cloud location
func WithGoogleLocation(location string) GoogleOption {
	return func(c *google.Config) {
		c.Location = location
	}
}

// WithGoogleBaseURL sets a custom base URL for Google AI
func WithGoogleBaseURL(baseURL string) GoogleOption {
	return func(c *google.Config) {
		c.BaseURL = baseURL
	}
}

// Simple Client Connections
func NewOpenAI(apiKey string) (*Client, error) {
	return NewAIClient().
		WithOpenAI(apiKey).
		WithDefaultProvider("openai").
		WithDefaultModel("gpt-4o-mini").
		Build()
}

func NewReplicate(apiKey string) (*Client, error) {
	return NewAIClient().
		WithReplicate(apiKey).
		WithDefaultProvider("replicate").
		WithDefaultModel("meta/meta-llama-3-8b-instruct").
		Build()
}

func NewGoogle(apiKey, projectID string) (*Client, error) {
	return NewAIClient().
		WithGoogle(apiKey, projectID).
		WithDefaultProvider("google").
		WithDefaultModel("gemini-2.5-flash").
		Build()
}
