package types

import "context"

// Provider defines the interface that all AI providers must implement
type Provider interface {
	// GetName returns the provider name
	GetName() string

	// Initialize sets up the provider with configuration
	Initialize(config Config) error

	// GetModels returns available models for this provider
	GetModels(ctx context.Context) ([]*Model, error)

	// Complete performs a completion request
	Complete(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error)

	// Stream performs a streaming completion request
	Stream(ctx context.Context, req *CompletionRequest, callback StreamCallback) error

	// EstimateTokens estimates the token count for given messages
	EstimateTokens(ctx context.Context, messages []*Message, model string) (int, error)

	// ValidateModel checks if a model is supported by this provider
	ValidateModel(model string) error

	// Close cleans up resources
	Close() error
}

// Config represents provider configuration interface
type Config interface {
	GetProvider() string
	Validate() error
}

// BaseConfig provides common configuration fields
type BaseConfig struct {
	Provider   string `json:"provider"`
	APIKey     string `json:"api_key"`
	BaseURL    string `json:"base_url,omitempty"`
	Timeout    int    `json:"timeout,omitempty"` // in seconds
	MaxRetries int    `json:"max_retries,omitempty"`
}

func (c *BaseConfig) GetProvider() string {
	return c.Provider
}

func (c *BaseConfig) Validate() error {
	if c.Provider == "" {
		return NewError(ErrCodeInvalidConfig, "provider is required", "")
	}
	if c.APIKey == "" {
		return NewError(ErrCodeInvalidConfig, "api_key is required", c.Provider)
	}
	return nil
}
