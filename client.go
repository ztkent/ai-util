package main

import (
	"context"
	"fmt"
	"sync"

	"github.com/ztkent/ai-util/types"
)

// Client is the main interface for interacting with AI providers
type Client struct {
	providers     map[string]types.Provider
	modelRegistry *types.ModelRegistry
	defaultConfig *ClientConfig
	mu            sync.RWMutex
}

// ClientConfig holds global client configuration
type ClientConfig struct {
	DefaultProvider    string                  `json:"default_provider,omitempty"`
	DefaultModel       string                  `json:"default_model,omitempty"`
	DefaultMaxTokens   int                     `json:"default_max_tokens,omitempty"`
	DefaultTemperature float64                 `json:"default_temperature,omitempty"`
	ProviderConfigs    map[string]types.Config `json:"provider_configs,omitempty"`
	Middleware         []Middleware            `json:"-"`
}

// Middleware defines the interface for request/response middleware
type Middleware interface {
	ProcessRequest(ctx context.Context, req *types.CompletionRequest) (*types.CompletionRequest, error)
	ProcessResponse(ctx context.Context, resp *types.CompletionResponse) (*types.CompletionResponse, error)
}

// LoggingMiddleware is an example middleware that logs requests and responses
type LoggingMiddleware struct{}

func (m *LoggingMiddleware) ProcessRequest(ctx context.Context, req *types.CompletionRequest) (*types.CompletionRequest, error) {
	fmt.Printf("Request: Model=%s, Messages=%d\n", req.Model, len(req.Messages))
	return req, nil
}

func (m *LoggingMiddleware) ProcessResponse(ctx context.Context, resp *types.CompletionResponse) (*types.CompletionResponse, error) {
	fmt.Printf("Response: Provider=%s, Tokens=%d\n", resp.Provider, resp.Usage.TotalTokens)
	return resp, nil
}

// NewClient creates a new AI client
func NewClient(config *ClientConfig) *Client {
	if config == nil {
		config = &ClientConfig{
			DefaultMaxTokens:   4096,
			DefaultTemperature: 0.7,
			ProviderConfigs:    make(map[string]types.Config),
		}
	}

	client := &Client{
		providers:     make(map[string]types.Provider),
		modelRegistry: types.NewModelRegistry(),
		defaultConfig: config,
	}

	return client
}

// RegisterProvider registers a new provider with the client
func (c *Client) RegisterProvider(provider types.Provider) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	providerName := provider.GetName()
	if _, exists := c.providers[providerName]; exists {
		return types.NewError(types.ErrCodeInvalidConfig,
			fmt.Sprintf("provider %s already registered", providerName), "")
	}

	// Initialize provider if config is available
	if config, exists := c.defaultConfig.ProviderConfigs[providerName]; exists {
		if err := provider.Initialize(config); err != nil {
			return types.WrapError(err, types.ErrCodeInvalidConfig, providerName)
		}
	}

	c.providers[providerName] = provider

	// Register models from this provider
	ctx := context.Background()
	models, err := provider.GetModels(ctx)
	if err != nil {
		// Log warning but don't fail registration
		fmt.Printf("Warning: failed to get models for provider %s: %v\n", providerName, err)
	} else {
		for _, model := range models {
			c.modelRegistry.Register(model)
		}
	}

	return nil
}

// GetProvider returns a provider by name
func (c *Client) GetProvider(name string) (types.Provider, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	provider, exists := c.providers[name]
	if !exists {
		return nil, types.NewError(types.ErrCodeInvalidConfig,
			fmt.Sprintf("provider %s not found", name), "")
	}

	return provider, nil
}

// GetModel returns a model by provider and ID
func (c *Client) GetModel(provider, id string) (*types.Model, error) {
	model, exists := c.modelRegistry.Get(provider, id)
	if !exists {
		return nil, types.NewError(types.ErrCodeModelNotFound,
			fmt.Sprintf("model %s not found for provider %s", id, provider), provider)
	}
	return model, nil
}

// ListModels returns all available models
func (c *Client) ListModels() []*types.Model {
	return c.modelRegistry.List()
}

// ListModelsByProvider returns models for a specific provider
func (c *Client) ListModelsByProvider(provider string) []*types.Model {
	return c.modelRegistry.GetByProvider(provider)
}

// Complete performs a completion request
func (c *Client) Complete(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error) {
	// Apply defaults
	if err := c.applyDefaults(req); err != nil {
		return nil, err
	}

	// Get provider for the model
	provider, err := c.getProviderForModel(req.Model)
	if err != nil {
		return nil, err
	}

	// Apply middleware to request
	processedReq := req
	for _, middleware := range c.defaultConfig.Middleware {
		processedReq, err = middleware.ProcessRequest(ctx, processedReq)
		if err != nil {
			return nil, types.WrapError(err, types.ErrCodeInvalidRequest, provider.GetName())
		}
	}

	// Perform completion
	resp, err := provider.Complete(ctx, processedReq)
	if err != nil {
		return nil, err
	}

	// Apply middleware to response
	for _, middleware := range c.defaultConfig.Middleware {
		resp, err = middleware.ProcessResponse(ctx, resp)
		if err != nil {
			return nil, types.WrapError(err, types.ErrCodeServerError, provider.GetName())
		}
	}

	return resp, nil
}

// Stream performs a streaming completion request
func (c *Client) Stream(ctx context.Context, req *types.CompletionRequest, callback types.StreamCallback) error {
	// Apply defaults
	if err := c.applyDefaults(req); err != nil {
		return err
	}

	// Get provider for the model
	provider, err := c.getProviderForModel(req.Model)
	if err != nil {
		return err
	}

	// Apply middleware to request
	processedReq := req
	for _, middleware := range c.defaultConfig.Middleware {
		processedReq, err = middleware.ProcessRequest(ctx, processedReq)
		if err != nil {
			return types.WrapError(err, types.ErrCodeInvalidRequest, provider.GetName())
		}
	}

	// Set stream flag
	processedReq.Stream = true

	// Perform streaming
	return provider.Stream(ctx, processedReq, callback)
}

// EstimateTokens estimates token count for messages and model
func (c *Client) EstimateTokens(ctx context.Context, messages []*types.Message, model string) (int, error) {
	provider, err := c.getProviderForModel(model)
	if err != nil {
		return 0, err
	}

	return provider.EstimateTokens(ctx, messages, model)
}

// Close closes all providers and cleans up resources
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	var errors []error
	for _, provider := range c.providers {
		if err := provider.Close(); err != nil {
			errors = append(errors, err)
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("errors closing providers: %v", errors)
	}

	return nil
}

// applyDefaults applies default configuration to the request
func (c *Client) applyDefaults(req *types.CompletionRequest) error {
	if req.Model == "" {
		if c.defaultConfig.DefaultModel == "" {
			return types.NewError(types.ErrCodeInvalidRequest, "model is required", "")
		}
		req.Model = c.defaultConfig.DefaultModel
	}

	if req.MaxTokens == 0 {
		req.MaxTokens = c.defaultConfig.DefaultMaxTokens
	}

	if req.Temperature == 0 {
		req.Temperature = c.defaultConfig.DefaultTemperature
	}

	return nil
}

// getProviderForModel determines which provider should handle the given model
func (c *Client) getProviderForModel(model string) (types.Provider, error) {
	// First try to find the model in registry
	for _, registeredModel := range c.modelRegistry.List() {
		if registeredModel.ID == model {
			return c.GetProvider(registeredModel.Provider)
		}
	}

	// Fallback to default provider if configured
	if c.defaultConfig.DefaultProvider != "" {
		return c.GetProvider(c.defaultConfig.DefaultProvider)
	}

	return nil, types.NewError(types.ErrCodeModelNotFound,
		fmt.Sprintf("no provider found for model %s", model), "")
}
