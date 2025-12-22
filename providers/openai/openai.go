package openai

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/sashabaranov/go-openai"
	"github.com/ztkent/ai-util/types"
)

// Provider implements the OpenAI provider
type Provider struct {
	client *openai.Client
	config *Config
}

// Config holds OpenAI-specific configuration
type Config struct {
	types.BaseConfig
	OrgID            string  `json:"org_id,omitempty"`
	Project          string  `json:"project,omitempty"`
	PresencePenalty  float32 `json:"presence_penalty,omitempty"`
	FrequencyPenalty float32 `json:"frequency_penalty,omitempty"`
	User             string  `json:"user,omitempty"`
}

// NewProvider creates a new OpenAI provider
func NewProvider() *Provider {
	return &Provider{}
}

// GetName returns the provider name
func (p *Provider) GetName() string {
	return "openai"
}

// Initialize sets up the provider with configuration
func (p *Provider) Initialize(config types.Config) error {
	openaiConfig, ok := config.(*Config)
	if !ok {
		return types.NewError(types.ErrCodeInvalidConfig, "invalid config type for OpenAI provider", "openai")
	}

	if err := openaiConfig.Validate(); err != nil {
		return err
	}

	clientConfig := openai.DefaultConfig(openaiConfig.APIKey)
	if openaiConfig.BaseURL != "" {
		clientConfig.BaseURL = openaiConfig.BaseURL
	}
	if openaiConfig.OrgID != "" {
		clientConfig.OrgID = openaiConfig.OrgID
	}

	p.client = openai.NewClientWithConfig(clientConfig)
	p.config = openaiConfig

	return nil
}

// GetModels returns available OpenAI models
func (p *Provider) GetModels(ctx context.Context) ([]*types.Model, error) {
	if p.client == nil {
		return nil, types.NewError(types.ErrCodeInvalidConfig, "provider not initialized", "openai")
	}

	response, err := p.client.ListModels(ctx)
	if err != nil {
		return nil, types.WrapError(err, types.ErrCodeServerError, "openai")
	}

	var models []*types.Model
	for _, model := range response.Models {
		aiModel := &types.Model{
			ID:           model.ID,
			Name:         model.ID,
			Provider:     "openai",
			Description:  fmt.Sprintf("OpenAI model: %s", model.ID),
			Capabilities: getModelCapabilities(model.ID),
		}

		// Set model-specific properties
		if maxTokens, ok := getModelMaxTokens(model.ID); ok {
			aiModel.MaxTokens = maxTokens
		}

		models = append(models, aiModel)
	}

	return models, nil
}

// Complete performs a completion request
func (p *Provider) Complete(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error) {
	if p.client == nil {
		return nil, types.NewError(types.ErrCodeInvalidConfig, "provider not initialized", "openai")
	}

	// Convert to OpenAI format
	openaiReq, err := p.convertRequest(req)
	if err != nil {
		return nil, err
	}

	resp, err := p.client.CreateChatCompletion(ctx, *openaiReq)
	if err != nil {
		return nil, types.WrapError(err, types.ErrCodeServerError, "openai")
	}

	// Convert response
	return p.convertResponse(&resp), nil
}

// Stream performs a streaming completion request
func (p *Provider) Stream(ctx context.Context, req *types.CompletionRequest, callback types.StreamCallback) error {
	if p.client == nil {
		return types.NewError(types.ErrCodeInvalidConfig, "provider not initialized", "openai")
	}

	// Convert to OpenAI format
	openaiReq, err := p.convertRequest(req)
	if err != nil {
		return err
	}
	openaiReq.Stream = true

	stream, err := p.client.CreateChatCompletionStream(ctx, *openaiReq)
	if err != nil {
		return types.WrapError(err, types.ErrCodeServerError, "openai")
	}
	defer stream.Close()

	for {
		response, err := stream.Recv()
		if err != nil {
			if err == io.EOF {
				break
			}
			return types.WrapError(err, types.ErrCodeServerError, "openai")
		}

		streamResp := p.convertStreamResponse(&response)
		if err := callback(ctx, streamResp); err != nil {
			return err
		}
	}

	return nil
}

// EstimateTokens estimates token count for messages
func (p *Provider) EstimateTokens(ctx context.Context, messages []*types.Message, model string) (int, error) {
	// This is a simplified estimation - in practice you'd use tiktoken or similar
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
		"gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
		"gpt-5", "o3-preview", "o3-mini",
		"o1-preview", "o1-mini", "gpt-4-1106-preview", "gpt-4-0125-preview",
	}

	for _, supported := range supportedModels {
		if model == supported {
			return nil
		}
	}

	return types.NewError(types.ErrCodeModelNotFound,
		fmt.Sprintf("model %s not supported by OpenAI provider", model), "openai")
}

// Close cleans up resources
func (p *Provider) Close() error {
	p.client = nil
	return nil
}

// convertRequest converts unified request to OpenAI format
func (p *Provider) convertRequest(req *types.CompletionRequest) (*openai.ChatCompletionRequest, error) {
	var messages []openai.ChatCompletionMessage

	for _, msg := range req.Messages {
		openaiMsg, err := p.convertMessage(msg)
		if err != nil {
			return nil, err
		}
		messages = append(messages, *openaiMsg)
	}

	openaiReq := &openai.ChatCompletionRequest{
		Model:       req.Model,
		Messages:    messages,
		MaxTokens:   req.MaxTokens,
		Temperature: float32(req.Temperature),
		TopP:        float32(req.TopP),
		Seed:        req.Seed,
		Stop:        req.Stop,
		Stream:      req.Stream,
		User:        p.config.User,
	}

	// Add tools if present
	if len(req.Tools) > 0 {
		tools := make([]openai.Tool, len(req.Tools))
		for i, tool := range req.Tools {
			tools[i] = openai.Tool{
				Type: openai.ToolType(tool.Type),
				Function: &openai.FunctionDefinition{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					Parameters:  tool.Function.Parameters,
				},
			}
		}
		openaiReq.Tools = tools
	}

	// Add response format if present
	if req.ResponseFormat != nil {
		openaiReq.ResponseFormat = &openai.ChatCompletionResponseFormat{
			Type: openai.ChatCompletionResponseFormatType(req.ResponseFormat.Type),
		}
	}

	return openaiReq, nil
}

// convertMessage converts unified message to OpenAI format
func (p *Provider) convertMessage(msg *types.Message) (*openai.ChatCompletionMessage, error) {
	openaiMsg := &openai.ChatCompletionMessage{
		Role: string(msg.Role),
	}

	// Handle simple text content
	if msg.TextData != "" {
		openaiMsg.Content = msg.TextData
		return openaiMsg, nil
	}

	// Handle structured content
	if len(msg.Content) > 0 {
		var parts []openai.ChatMessagePart
		for _, content := range msg.Content {
			switch c := content.(type) {
			case types.TextContent:
				parts = append(parts, openai.ChatMessagePart{
					Type: openai.ChatMessagePartTypeText,
					Text: c.Text,
				})
			case types.ImageContent:
				imageURL := &openai.ChatMessageImageURL{
					Detail: openai.ImageURLDetail(c.Detail),
				}
				if c.URL != "" {
					imageURL.URL = c.URL
				} else if c.Base64 != "" {
					imageURL.URL = fmt.Sprintf("data:image/jpeg;base64,%s", c.Base64)
				}
				parts = append(parts, openai.ChatMessagePart{
					Type:     openai.ChatMessagePartTypeImageURL,
					ImageURL: imageURL,
				})
			}
		}
		openaiMsg.MultiContent = parts
	}

	// Handle tool calls
	if len(msg.ToolCalls) > 0 {
		var toolCalls []openai.ToolCall
		for _, tc := range msg.ToolCalls {
			toolCalls = append(toolCalls, openai.ToolCall{
				ID:   tc.ID,
				Type: openai.ToolType(tc.Type),
				Function: openai.FunctionCall{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			})
		}
		openaiMsg.ToolCalls = toolCalls
	}

	// Handle tool results
	if msg.ToolResult != nil {
		openaiMsg.ToolCallID = msg.ToolResult.ToolCallID
		openaiMsg.Content = msg.ToolResult.Content
	}

	return openaiMsg, nil
}

// convertResponse converts OpenAI response to unified format
func (p *Provider) convertResponse(resp *openai.ChatCompletionResponse) *types.CompletionResponse {
	var message *types.Message
	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]
		message = &types.Message{
			Role:     types.Role(choice.Message.Role),
			TextData: choice.Message.Content,
		}

		// Handle tool calls
		if len(choice.Message.ToolCalls) > 0 {
			var toolCalls []types.ToolCall
			for _, tc := range choice.Message.ToolCalls {
				toolCalls = append(toolCalls, types.ToolCall{
					ID:   tc.ID,
					Type: string(tc.Type),
					Function: types.ToolCallFunction{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				})
			}
			message.ToolCalls = toolCalls
		}
	}

	usage := &types.Usage{
		PromptTokens:     resp.Usage.PromptTokens,
		CompletionTokens: resp.Usage.CompletionTokens,
		TotalTokens:      resp.Usage.TotalTokens,
	}

	return &types.CompletionResponse{
		ID:           resp.ID,
		Model:        resp.Model,
		Provider:     "openai",
		Message:      message,
		FinishReason: string(resp.Choices[0].FinishReason),
		Usage:        usage,
		Created:      int64(resp.Created),
	}
}

// convertStreamResponse converts OpenAI stream response to unified format
func (p *Provider) convertStreamResponse(resp *openai.ChatCompletionStreamResponse) *types.StreamResponse {
	var delta *types.Message
	var finishReason string

	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]
		delta = &types.Message{
			Role:     types.Role(choice.Delta.Role),
			TextData: choice.Delta.Content,
		}

		// Handle tool calls in delta
		if len(choice.Delta.ToolCalls) > 0 {
			var toolCalls []types.ToolCall
			for _, tc := range choice.Delta.ToolCalls {
				toolCalls = append(toolCalls, types.ToolCall{
					ID:   tc.ID,
					Type: string(tc.Type),
					Function: types.ToolCallFunction{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				})
			}
			delta.ToolCalls = toolCalls
		}

		finishReason = string(choice.FinishReason)
	}

	var usage *types.Usage
	if resp.Usage != nil {
		usage = &types.Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		}
	}

	return &types.StreamResponse{
		ID:           resp.ID,
		Model:        resp.Model,
		Provider:     "openai",
		Delta:        delta,
		FinishReason: finishReason,
		Usage:        usage,
	}
}

// getModelCapabilities returns capabilities for a given model
func getModelCapabilities(modelID string) []string {
	capabilities := []string{string(types.CapabilityChat), string(types.CapabilityStreaming)}

	// Add tools capability for newer models
	capabilities = append(capabilities, string(types.CapabilityTools))

	// Add JSON capability for supported models
	if strings.Contains(modelID, "gpt-4") || strings.Contains(modelID, "gpt-5") || strings.Contains(modelID, "o3") || strings.Contains(modelID, "o1") {
		capabilities = append(capabilities, string(types.CapabilityJSON))
	}

	return capabilities
}

// getModelMaxTokens returns max tokens for known models
func getModelMaxTokens(modelID string) (int, bool) {
	maxTokens := map[string]int{
		"gpt-4":         8192,
		"gpt-4-turbo":   128000,
		"gpt-4o":        128000,
		"gpt-4o-mini":   128000,
		"gpt-5":         200000,
		"o1-preview":    128000,
		"o1-mini":       128000,
		"o3-preview":    200000,
		"o3-mini":       200000,
	}

	tokens, exists := maxTokens[modelID]
	return tokens, exists
}
