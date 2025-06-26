package replicate

import (
	"context"
	"fmt"
	"strings"

	"github.com/replicate/replicate-go"
	"github.com/ztkent/ai-util/types"
)

// Provider implements the Replicate provider
type Provider struct {
	client *replicate.Client
	config *Config
}

// Config holds Replicate-specific configuration
type Config struct {
	types.BaseConfig
	WebhookURL  string                 `json:"webhook_url,omitempty"`
	ExtraInputs map[string]interface{} `json:"extra_inputs,omitempty"`
}

// NewProvider creates a new Replicate provider
func NewProvider() *Provider {
	return &Provider{}
}

// GetName returns the provider name
func (p *Provider) GetName() string {
	return "replicate"
}

// Initialize sets up the provider with configuration
func (p *Provider) Initialize(config types.Config) error {
	replicateConfig, ok := config.(*Config)
	if !ok {
		return types.NewError(types.ErrCodeInvalidConfig, "invalid config type for Replicate provider", "replicate")
	}

	if err := replicateConfig.Validate(); err != nil {
		return err
	}

	client, err := replicate.NewClient(replicate.WithToken(replicateConfig.APIKey))
	if err != nil {
		return types.WrapError(err, types.ErrCodeInvalidConfig, "replicate")
	}

	p.client = client
	p.config = replicateConfig

	return nil
}

// GetModels returns available Replicate models
func (p *Provider) GetModels(ctx context.Context) ([]*types.Model, error) {
	if p.client == nil {
		return nil, types.NewError(types.ErrCodeInvalidConfig, "provider not initialized", "replicate")
	}

	// For Replicate, we'll return a curated list of popular chat models
	// In practice, you might want to query the Replicate API for available models
	models := []*types.Model{
		{
			ID:          "meta/meta-llama-3-8b-instruct",
			Name:        "Meta Llama 3 8B Instruct",
			Provider:    "replicate",
			Description: "Meta's Llama 3 8B parameter instruction-tuned model",
			MaxTokens:   8192,
			Capabilities: []string{
				string(types.CapabilityChat),
				string(types.CapabilityStreaming),
			},
		},
		{
			ID:          "meta/meta-llama-3-70b-instruct",
			Name:        "Meta Llama 3 70B Instruct",
			Provider:    "replicate",
			Description: "Meta's Llama 3 70B parameter instruction-tuned model",
			MaxTokens:   8192,
			Capabilities: []string{
				string(types.CapabilityChat),
				string(types.CapabilityStreaming),
			},
		},
		{
			ID:          "mistralai/mistral-7b-instruct-v0.2",
			Name:        "Mistral 7B Instruct",
			Provider:    "replicate",
			Description: "Mistral AI's 7B parameter instruction-tuned model",
			MaxTokens:   32768,
			Capabilities: []string{
				string(types.CapabilityChat),
				string(types.CapabilityStreaming),
			},
		},
		{
			ID:          "mistralai/mixtral-8x7b-instruct-v0.1",
			Name:        "Mixtral 8x7B Instruct",
			Provider:    "replicate",
			Description: "Mistral AI's Mixtral 8x7B parameter mixture of experts model",
			MaxTokens:   32768,
			Capabilities: []string{
				string(types.CapabilityChat),
				string(types.CapabilityStreaming),
			},
		},
	}

	return models, nil
}

// Complete performs a completion request
func (p *Provider) Complete(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error) {
	if p.client == nil {
		return nil, types.NewError(types.ErrCodeInvalidConfig, "provider not initialized", "replicate")
	}

	// Convert request to Replicate format
	input, err := p.convertRequest(req)
	if err != nil {
		return nil, err
	}

	// Create webhook if configured
	var webhook *replicate.Webhook
	if p.config.WebhookURL != "" {
		webhook = &replicate.Webhook{
			URL: p.config.WebhookURL,
		}
	}

	// Run prediction
	prediction, err := p.client.CreatePrediction(ctx, req.Model, input, webhook, false)
	if err != nil {
		return nil, types.WrapError(err, types.ErrCodeServerError, "replicate")
	}

	// Wait for completion
	err = p.client.Wait(ctx, prediction)
	if err != nil {
		return nil, types.WrapError(err, types.ErrCodeServerError, "replicate")
	}

	// Convert response
	return p.convertResponse(prediction), nil
}

// Stream performs a streaming completion request
func (p *Provider) Stream(ctx context.Context, req *types.CompletionRequest, callback types.StreamCallback) error {
	if p.client == nil {
		return types.NewError(types.ErrCodeInvalidConfig, "provider not initialized", "replicate")
	}

	// For now, we'll implement streaming by polling the prediction
	// Replicate's streaming API is different and would need specific implementation
	resp, err := p.Complete(ctx, req)
	if err != nil {
		return err
	}

	// Simulate streaming by sending the complete response as a stream event
	streamResp := &types.StreamResponse{
		ID:           resp.ID,
		Model:        resp.Model,
		Provider:     "replicate",
		Delta:        resp.Message,
		FinishReason: resp.FinishReason,
		Usage:        resp.Usage,
	}

	return callback(ctx, streamResp)
}

// EstimateTokens estimates token count for messages
func (p *Provider) EstimateTokens(ctx context.Context, messages []*types.Message, model string) (int, error) {
	// Simple estimation for Replicate models
	totalTokens := 0
	for _, msg := range messages {
		text := msg.GetText()
		// Rough estimation: ~4 characters per token
		totalTokens += len(text) / 4
	}
	return totalTokens, nil
}

// ValidateModel checks if a model is supported
func (p *Provider) ValidateModel(model string) error {
	supportedModels := []string{
		"meta/meta-llama-3-8b-instruct",
		"meta/meta-llama-3-70b-instruct",
		"mistralai/mistral-7b-instruct-v0.2",
		"mistralai/mixtral-8x7b-instruct-v0.1",
	}

	for _, supported := range supportedModels {
		if model == supported {
			return nil
		}
	}

	return types.NewError(types.ErrCodeModelNotFound,
		fmt.Sprintf("model %s not supported by Replicate provider", model), "replicate")
}

// Close cleans up resources
func (p *Provider) Close() error {
	p.client = nil
	return nil
}

// convertRequest converts unified request to Replicate format
func (p *Provider) convertRequest(req *types.CompletionRequest) (map[string]interface{}, error) {
	// Build prompt from messages
	prompt := p.buildPromptFromMessages(req.Messages)

	input := map[string]interface{}{
		"prompt": prompt,
	}

	// Add generation parameters
	if req.MaxTokens > 0 {
		input["max_new_tokens"] = req.MaxTokens
	}
	if req.Temperature > 0 {
		input["temperature"] = req.Temperature
	}
	if req.TopP > 0 {
		input["top_p"] = req.TopP
	}
	if req.TopK > 0 {
		input["top_k"] = req.TopK
	}
	if len(req.Stop) > 0 {
		input["stop_sequences"] = strings.Join(req.Stop, ",")
	}

	// Add extra inputs from config
	for k, v := range p.config.ExtraInputs {
		input[k] = v
	}

	return input, nil
}

// buildPromptFromMessages converts messages to a single prompt string
func (p *Provider) buildPromptFromMessages(messages []*types.Message) string {
	var parts []string

	for _, msg := range messages {
		text := msg.GetText()
		if text == "" {
			continue
		}

		switch msg.Role {
		case types.RoleSystem:
			parts = append(parts, fmt.Sprintf("System: %s", text))
		case types.RoleUser:
			parts = append(parts, fmt.Sprintf("Human: %s", text))
		case types.RoleAssistant:
			parts = append(parts, fmt.Sprintf("Assistant: %s", text))
		}
	}

	// Add final prompt for assistant response
	return strings.Join(parts, "\n\n") + "\n\nAssistant: "
}

// convertResponse converts Replicate prediction to unified format
func (p *Provider) convertResponse(prediction *replicate.Prediction) *types.CompletionResponse {
	var content string
	if prediction.Output != nil {
		if outputSlice, ok := prediction.Output.([]interface{}); ok {
			var parts []string
			for _, part := range outputSlice {
				if str, ok := part.(string); ok {
					parts = append(parts, str)
				}
			}
			content = strings.Join(parts, "")
		} else if str, ok := prediction.Output.(string); ok {
			content = str
		}
	}

	message := &types.Message{
		Role:     types.RoleAssistant,
		TextData: content,
	}

	// Estimate usage (Replicate doesn't provide token counts)
	usage := &types.Usage{
		CompletionTokens: len(content) / 4, // Rough estimation
		TotalTokens:      len(content) / 4,
	}

	finishReason := "stop"
	switch prediction.Status {
	case "failed":
		finishReason = "error"
	case "canceled":
		finishReason = "cancelled"
	}

	return &types.CompletionResponse{
		ID:           prediction.ID,
		Model:        prediction.Model,
		Provider:     "replicate",
		Message:      message,
		FinishReason: finishReason,
		Usage:        usage,
	}
}
