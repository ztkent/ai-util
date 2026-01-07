package google

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ztkent/ai-util/types"
	"google.golang.org/genai"
)

// Provider implements the Google AI provider
type Provider struct {
	config *Config
	client *genai.Client
}

// Config holds Google AI-specific configuration
type Config struct {
	types.BaseConfig
	ProjectID string `json:"project_id,omitempty"`
	Location  string `json:"location,omitempty"`
}

// NewProvider creates a new Google AI provider
func NewProvider() *Provider {
	return &Provider{}
}

// GetName returns the provider name
func (p *Provider) GetName() string {
	return "google"
}

// Initialize sets up the provider with configuration
func (p *Provider) Initialize(config types.Config) error {
	googleConfig, ok := config.(*Config)
	if !ok {
		return types.NewError(types.ErrCodeInvalidConfig, "invalid config type for Google provider", "google")
	}

	if err := googleConfig.Validate(); err != nil {
		return err
	}

	// Initialize Google AI client
	ctx := context.Background()
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  googleConfig.APIKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return types.WrapError(err, types.ErrCodeAuthentication, "google")
	}

	p.config = googleConfig
	p.client = client

	return nil
}

// GetModels returns available Google AI models
func (p *Provider) GetModels(ctx context.Context) ([]*types.Model, error) {
	if p.config == nil {
		return nil, types.NewError(types.ErrCodeInvalidConfig, "provider not initialized", "google")
	}

	// Return a list of current Google AI models based on official documentation
	models := []*types.Model{
		// Gemini 3.0 series - Next generation reasoning
		{
			ID:          "gemini-3-pro-preview",
			Name:        "Gemini 3 Pro",
			Provider:    "google",
			Description: "The most capable AI model, built for the future of reasoning and coding",
			MaxTokens:   4000000,
			Capabilities: []string{
				string(types.CapabilityChat),
				string(types.CapabilityStreaming),
				string(types.CapabilityTools),
				string(types.CapabilityVision),
				string(types.CapabilityAudio),
				string(types.CapabilityVideo),
				string(types.CapabilityThinking),
				string(types.CapabilityJSON),
			},
		},
		{
			ID:          "gemini-3-flash-preview",
			Name:        "Gemini 3 Flash Preview",
			Provider:    "google",
			Description: "Ultra-fast, low latency model with advanced reasoning capabilities",
			MaxTokens:   2000000,
			Capabilities: []string{
				string(types.CapabilityChat),
				string(types.CapabilityStreaming),
				string(types.CapabilityTools),
				string(types.CapabilityVision),
				string(types.CapabilityAudio),
				string(types.CapabilityVideo),
				string(types.CapabilityThinking),
				string(types.CapabilityJSON),
			},
		},
		// Gemini 2.5 series - Latest thinking models
		{
			ID:          "gemini-2.5-pro",
			Name:        "Gemini 2.5 Pro",
			Provider:    "google",
			Description: "Most powerful thinking model with maximum response accuracy and state-of-the-art performance",
			MaxTokens:   2000000,
			Capabilities: []string{
				string(types.CapabilityChat),
				string(types.CapabilityStreaming),
				string(types.CapabilityTools),
				string(types.CapabilityVision),
				string(types.CapabilityAudio),
				string(types.CapabilityVideo),
				string(types.CapabilityThinking),
				string(types.CapabilityJSON),
			},
		},
		{
			ID:          "gemini-2.5-flash",
			Name:        "Gemini 2.5 Flash",
			Provider:    "google",
			Description: "Best model in terms of price-performance with adaptive thinking capabilities",
			MaxTokens:   1000000,
			Capabilities: []string{
				string(types.CapabilityChat),
				string(types.CapabilityStreaming),
				string(types.CapabilityTools),
				string(types.CapabilityVision),
				string(types.CapabilityAudio),
				string(types.CapabilityVideo),
				string(types.CapabilityThinking),
				string(types.CapabilityJSON),
			},
		},
		{
			ID:          "gemini-2.5-flash-lite",
			Name:        "Gemini 2.5 Flash-Lite",
			Provider:    "google",
			Description: "Most cost-efficient model optimized for high throughput and low latency",
			MaxTokens:   1000000,
			Capabilities: []string{
				string(types.CapabilityChat),
				string(types.CapabilityStreaming),
				string(types.CapabilityTools),
				string(types.CapabilityVision),
				string(types.CapabilityAudio),
				string(types.CapabilityVideo),
				string(types.CapabilityJSON),
			},
		},
		{
			ID:          "gemini-2.5-flash-preview-tts",
			Name:        "Gemini 2.5 Flash Preview TTS",
			Provider:    "google",
			Description: "Low latency, controllable text-to-speech audio generation",
			MaxTokens:   1000000,
			Capabilities: []string{
				string(types.CapabilityTTS),
				string(types.CapabilityJSON),
			},
		},
		{
			ID:          "gemini-2.5-pro-preview-tts",
			Name:        "Gemini 2.5 Pro Preview TTS",
			Provider:    "google",
			Description: "High-quality text-to-speech with single and multi-speaker support",
			MaxTokens:   2000000,
			Capabilities: []string{
				string(types.CapabilityTTS),
				string(types.CapabilityJSON),
			},
		},
		// Live interaction models
		{
			ID:          "gemini-2.5-flash-live",
			Name:        "Gemini 2.5 Flash Live",
			Provider:    "google",
			Description: "Low-latency bidirectional voice and video interactions",
			MaxTokens:   1000000,
			Capabilities: []string{
				string(types.CapabilityLive),
				string(types.CapabilityAudio),
				string(types.CapabilityVideo),
				string(types.CapabilityStreaming),
			},
		},
		// Gemma 3 series
		{
			ID:          "gemma-3-27b-it",
			Name:        "Gemma 3 27B IT",
			Provider:    "google",
			Description: "Best for complex reasoning and chat",
			MaxTokens:   8192,
			Capabilities: []string{
				string(types.CapabilityChat),
				string(types.CapabilityStreaming),
				string(types.CapabilityTools),
				string(types.CapabilityJSON),
			},
		},
		{
			ID:          "gemma-3-12b-it",
			Name:        "Gemma 3 12B IT",
			Provider:    "google",
			Description: "High performance for laptops/desktops",
			MaxTokens:   8192,
			Capabilities: []string{
				string(types.CapabilityChat),
				string(types.CapabilityStreaming),
				string(types.CapabilityTools),
				string(types.CapabilityJSON),
			},
		},
		{
			ID:          "gemma-3-4b-it",
			Name:        "Gemma 3 4B IT",
			Provider:    "google",
			Description: "Balanced for efficiency and mobile",
			MaxTokens:   8192,
			Capabilities: []string{
				string(types.CapabilityChat),
				string(types.CapabilityStreaming),
				string(types.CapabilityTools),
				string(types.CapabilityJSON),
			},
		},
		{
			ID:          "gemma-3-1b-it",
			Name:        "Gemma 3 1B IT",
			Provider:    "google",
			Description: "Ultra-efficient for text-only tasks",
			MaxTokens:   8192,
			Capabilities: []string{
				string(types.CapabilityChat),
				string(types.CapabilityStreaming),
				string(types.CapabilityTools),
				string(types.CapabilityJSON),
			},
		},
		// Embedding models
		{
			ID:          "text-embedding-004",
			Name:        "Text Embedding 004",
			Provider:    "google",
			Description: "Latest text embedding model for measuring relatedness of text strings",
			MaxTokens:   8192,
			Capabilities: []string{
				"embedding",
			},
		},
		{
			ID:          "gemini-embedding-exp",
			Name:        "Gemini Embedding Experimental",
			Provider:    "google",
			Description: "Experimental embedding model with enhanced capabilities",
			MaxTokens:   8192,
			Capabilities: []string{
				"embedding",
			},
		},
		// Image and video generation models
		{
			ID:          "imagen-4.0-generate-preview",
			Name:        "Imagen 4",
			Provider:    "google",
			Description: "Most up-to-date image generation model with high quality outputs",
			MaxTokens:   1024,
			Capabilities: []string{
				string(types.CapabilityImage),
				string(types.CapabilityJSON),
			},
		},
		{
			ID:          "imagen-3.0-generate-002",
			Name:        "Imagen 3",
			Provider:    "google",
			Description: "High quality image generation model",
			MaxTokens:   1024,
			Capabilities: []string{
				string(types.CapabilityImage),
				string(types.CapabilityJSON),
			},
		},
		{
			ID:          "veo-2.0-generate-001",
			Name:        "Veo 2",
			Provider:    "google",
			Description: "High quality video generation from text and images",
			MaxTokens:   1024,
			Capabilities: []string{
				"video_generation",
				string(types.CapabilityJSON),
			},
		},
		{
			ID:          "veo-3.0-generate-001",
			Name:        "Veo 3",
			Provider:    "google",
			Description: "High quality video generation from text and images",
			MaxTokens:   1024,
			Capabilities: []string{
				"video_generation",
				string(types.CapabilityJSON),
			},
		},
	}

	return models, nil
}

// Complete performs a completion request
func (p *Provider) Complete(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error) {
	if p.config == nil {
		return nil, types.NewError(types.ErrCodeInvalidConfig, "provider not initialized", "google")
	}

	if p.client == nil {
		return nil, types.NewError(types.ErrCodeInvalidConfig, "Google AI client not initialized", "google")
	}

	// Convert messages to content format
	var contents []*genai.Content
	for _, msg := range req.Messages {
		text := msg.GetText()
		if text != "" {
			var role genai.Role
			switch msg.Role {
			case types.RoleUser:
				role = genai.RoleUser
			case types.RoleAssistant:
				role = genai.RoleModel
			case types.RoleSystem:
				role = genai.RoleUser // System messages are treated as user messages in Gemini
			default:
				role = genai.RoleUser
			}

			content := genai.NewContentFromText(text, role)
			contents = append(contents, content)
		}
	}

	// Create generation config
	var config *genai.GenerateContentConfig
	needsConfig := req.MaxTokens > 0 || req.Temperature > 0 || req.TopP > 0 || req.TopK > 0 || len(req.Tools) > 0 || len(req.GroundingTools) > 0 || req.ResponseFormat != nil
	if needsConfig {
		config = &genai.GenerateContentConfig{}

		// Set generation parameters
		if req.MaxTokens > 0 {
			config.MaxOutputTokens = int32(req.MaxTokens)
		}
		if req.Temperature > 0 {
			temp := float32(req.Temperature)
			config.Temperature = &temp
		}
		if req.TopP > 0 {
			topP := float32(req.TopP)
			config.TopP = &topP
		}
		if req.TopK > 0 {
			topK := float32(req.TopK)
			config.TopK = &topK
		}

		// Add function tools if present
		var tools []*genai.Tool
		if len(req.Tools) > 0 {
			for _, tool := range req.Tools {
				funcDecl := &genai.FunctionDeclaration{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					Parameters:  convertJSONSchemaToGeminiSchema(tool.Function.Parameters),
				}
				tools = append(tools, &genai.Tool{
					FunctionDeclarations: []*genai.FunctionDeclaration{funcDecl},
				})
			}
		}

		// Add grounding tools if present (Google-specific: URL context, Google Search)
		if len(req.GroundingTools) > 0 {
			for _, gt := range req.GroundingTools {
				switch gt.Type {
				case types.GroundingToolURLContext:
					tools = append(tools, &genai.Tool{
						URLContext: &genai.URLContext{},
					})
				case types.GroundingToolGoogleSearch:
					tools = append(tools, &genai.Tool{
						GoogleSearch: &genai.GoogleSearch{},
					})
				}
			}
		}

		if req.ThinkingConfig != nil {
			config.ThinkingConfig = &genai.ThinkingConfig{
				IncludeThoughts: req.ThinkingConfig.IncludeThoughts,
				ThinkingBudget:  req.ThinkingConfig.ThinkingBudget,
			}
		}

		if len(tools) > 0 {
			config.Tools = tools
		}

		// Set JSON response format if requested
		if req.ResponseFormat != nil && req.ResponseFormat.Type == "json_object" {
			config.ResponseMIMEType = "application/json"
			if req.ResponseFormat.Schema != nil {
				config.ResponseSchema = convertJSONSchemaToGeminiSchema(req.ResponseFormat.Schema)
			}
		}
	}
	// Generate content using the correct API
	result, err := p.client.Models.GenerateContent(
		ctx,
		req.Model,
		contents,
		config,
	)
	if err != nil {
		return nil, types.WrapError(err, types.ErrCodeServerError, "google")
	}

	// Extract response text
	responseText := result.Text()

	// Create the message
	message := &types.Message{
		Role:     types.RoleAssistant,
		TextData: responseText,
	}

	// Handle tool calls if present
	if len(result.Candidates) > 0 {
		toolCalls := p.handleToolCalls(result.Candidates)
		if len(toolCalls) > 0 {
			message.ToolCalls = toolCalls
		}
	}

	// Convert usage information if available
	var usage *types.Usage
	if result.UsageMetadata != nil {
		usage = &types.Usage{
			PromptTokens:     int(result.UsageMetadata.PromptTokenCount),
			CompletionTokens: int(result.UsageMetadata.CandidatesTokenCount),
			TotalTokens:      int(result.UsageMetadata.TotalTokenCount),
		}
	}

	// Determine finish reason
	finishReason := "stop"
	if len(result.Candidates) > 0 && len(result.Candidates[0].FinishReason) > 0 {
		finishReason = string(result.Candidates[0].FinishReason)
	}

	// Generate a simple ID
	responseID := fmt.Sprintf("google-completion-%d", len(responseText))
	if usage != nil {
		responseID = fmt.Sprintf("google-%d", usage.TotalTokens)
	}

	return &types.CompletionResponse{
		ID:           responseID,
		Model:        req.Model,
		Provider:     "google",
		Message:      message,
		FinishReason: finishReason,
		Usage:        usage,
	}, nil
}

// Stream performs a streaming completion request
func (p *Provider) Stream(ctx context.Context, req *types.CompletionRequest, callback types.StreamCallback) error {
	if p.config == nil {
		return types.NewError(types.ErrCodeInvalidConfig, "provider not initialized", "google")
	}

	if p.client == nil {
		return types.NewError(types.ErrCodeInvalidConfig, "Google AI client not initialized", "google")
	}

	// Convert messages to content format for streaming
	var contents []*genai.Content
	for _, msg := range req.Messages {
		text := msg.GetText()
		if text != "" {
			var role genai.Role
			switch msg.Role {
			case types.RoleUser:
				role = genai.RoleUser
			case types.RoleAssistant:
				role = genai.RoleModel
			case types.RoleSystem:
				role = genai.RoleUser // System messages are treated as user messages in Gemini
			default:
				role = genai.RoleUser
			}

			content := genai.NewContentFromText(text, role)
			contents = append(contents, content)
		}
	}

	// Create generation config
	var config *genai.GenerateContentConfig
	needsConfig := req.MaxTokens > 0 || req.Temperature > 0 || req.TopP > 0 || req.TopK > 0 || len(req.Tools) > 0 || len(req.GroundingTools) > 0 || req.ResponseFormat != nil
	if needsConfig {
		config = &genai.GenerateContentConfig{}

		// Set generation parameters
		if req.MaxTokens > 0 {
			config.MaxOutputTokens = int32(req.MaxTokens)
		}
		if req.Temperature > 0 {
			temp := float32(req.Temperature)
			config.Temperature = &temp
		}
		if req.TopP > 0 {
			topP := float32(req.TopP)
			config.TopP = &topP
		}
		if req.TopK > 0 {
			topK := float32(req.TopK)
			config.TopK = &topK
		}

		// Add function tools if present
		var tools []*genai.Tool
		if len(req.Tools) > 0 {
			for _, tool := range req.Tools {
				funcDecl := &genai.FunctionDeclaration{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					Parameters:  convertJSONSchemaToGeminiSchema(tool.Function.Parameters),
				}
				tools = append(tools, &genai.Tool{
					FunctionDeclarations: []*genai.FunctionDeclaration{funcDecl},
				})
			}
		}

		// Add grounding tools if present (Google-specific: URL context, Google Search)
		if len(req.GroundingTools) > 0 {
			for _, gt := range req.GroundingTools {
				switch gt.Type {
				case types.GroundingToolURLContext:
					tools = append(tools, &genai.Tool{
						URLContext: &genai.URLContext{},
					})
				case types.GroundingToolGoogleSearch:
					tools = append(tools, &genai.Tool{
						GoogleSearch: &genai.GoogleSearch{},
					})
				}
			}
		}

		if len(tools) > 0 {
			config.Tools = tools
		}

		// Set JSON response format if requested
		if req.ResponseFormat != nil && req.ResponseFormat.Type == "json_object" {
			config.ResponseMIMEType = "application/json"
			if req.ResponseFormat.Schema != nil {
				config.ResponseSchema = convertJSONSchemaToGeminiSchema(req.ResponseFormat.Schema)
			}
		}

		// Disable thinking for faster responses by default
		// thinkingBudget := int32(0)
		// config.ThinkingConfig = &genai.ThinkingConfig{
		// 	ThinkingBudget: &thinkingBudget,
		// }
	} else {
		// Default config with thinking disabled
		// thinkingBudget := int32(0)
		// config = &genai.GenerateContentConfig{
		// 	ThinkingConfig: &genai.ThinkingConfig{
		// 		ThinkingBudget: &thinkingBudget,
		// 	},
		// }
	}

	// Generate streaming content using the iterator
	responseID := fmt.Sprintf("google-stream-%d", len(req.Messages))
	var fullResponse strings.Builder
	var lastUsage *types.Usage

	// Use the streaming API
	stream := p.client.Models.GenerateContentStream(ctx, req.Model, contents, config)

	for response, err := range stream {
		if err != nil {
			return types.WrapError(err, types.ErrCodeServerError, "google")
		}

		// Extract text from this chunk
		chunkText := response.Text()
		fullResponse.WriteString(chunkText)

		// Convert usage information if available
		if response.UsageMetadata != nil {
			lastUsage = &types.Usage{
				PromptTokens:     int(response.UsageMetadata.PromptTokenCount),
				CompletionTokens: int(response.UsageMetadata.CandidatesTokenCount),
				TotalTokens:      int(response.UsageMetadata.TotalTokenCount),
			}
		}

		// Determine finish reason
		finishReason := ""
		if len(response.Candidates) > 0 && len(response.Candidates[0].FinishReason) > 0 {
			finishReason = string(response.Candidates[0].FinishReason)
		}

		// Send chunk to callback
		streamResp := &types.StreamResponse{
			ID:       responseID,
			Model:    req.Model,
			Provider: "google",
			Delta: &types.Message{
				Role:     types.RoleAssistant,
				TextData: chunkText,
			},
			FinishReason: finishReason,
			Usage:        lastUsage,
		}

		if err := callback(ctx, streamResp); err != nil {
			return err
		}

		// If we have a finish reason, this is the last chunk
		if finishReason != "" {
			break
		}
	}

	return nil
}

// EstimateTokens estimates token count for messages
func (p *Provider) EstimateTokens(ctx context.Context, messages []*types.Message, model string) (int, error) {
	// Simple estimation for Google models
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
		// Gemini 3.0 series
		"gemini-3-pro-preview",
		"gemini-3-flash-preview",
		// Gemini 2.5 series
		"gemini-2.5-pro",
		"gemini-2.5-flash",
		"gemini-2.5-flash-lite",
		"gemini-2.5-flash-preview-tts",
		"gemini-2.5-pro-preview-tts",
		"gemini-2.5-flash-live",
		// Gemma 3 series
		"gemma-3-27b-it",
		"gemma-3-12b-it",
		"gemma-3-4b-it",
		"gemma-3-1b-it",
		// Embedding models
		"text-embedding-004",
		// Image and video generation
		"imagen-4.0-generate-preview",
		"imagen-3.0-generate-002",
		"veo-3.0-generate-001",
		"veo-2.0-generate-001",
	}

	for _, supported := range supportedModels {
		if model == supported {
			return nil
		}
	}

	return types.NewError(types.ErrCodeModelNotFound,
		fmt.Sprintf("model %s not supported by Google provider", model), "google")
}

// Close cleans up resources
func (p *Provider) Close() error {
	p.client = nil
	return nil
}

// Validate validates Google-specific configuration
func (c *Config) Validate() error {
	if err := c.BaseConfig.Validate(); err != nil {
		return err
	}

	// For Gemini API, ProjectID is optional
	// If using Vertex AI, ProjectID would be required
	if c.Location == "" {
		c.Location = "us-central1" // Default location
	}

	return nil
}

// convertJSONSchemaToGeminiSchema converts a JSON schema to Gemini schema format
func convertJSONSchemaToGeminiSchema(schema interface{}) *genai.Schema {
	if schema == nil {
		return nil
	}

	schemaMap, ok := schema.(map[string]interface{})
	if !ok {
		return nil
	}

	geminiSchema := &genai.Schema{}

	if typeVal, exists := schemaMap["type"]; exists {
		if typeStr, ok := typeVal.(string); ok {
			switch typeStr {
			case "string":
				geminiSchema.Type = genai.TypeString
			case "number":
				geminiSchema.Type = genai.TypeNumber
			case "integer":
				geminiSchema.Type = genai.TypeInteger
			case "boolean":
				geminiSchema.Type = genai.TypeBoolean
			case "array":
				geminiSchema.Type = genai.TypeArray
			case "object":
				geminiSchema.Type = genai.TypeObject
			}
		}
	}

	if description, exists := schemaMap["description"]; exists {
		if descStr, ok := description.(string); ok {
			geminiSchema.Description = descStr
		}
	}

	// Handle enum values
	if enumVal, exists := schemaMap["enum"]; exists {
		if enumSlice, ok := enumVal.([]interface{}); ok {
			enumStrings := make([]string, len(enumSlice))
			for i, v := range enumSlice {
				if str, ok := v.(string); ok {
					enumStrings[i] = str
				}
			}
			geminiSchema.Enum = enumStrings
		}
	}

	// Handle array items
	if items, exists := schemaMap["items"]; exists {
		geminiSchema.Items = convertJSONSchemaToGeminiSchema(items)
	}

	// Handle object properties
	if properties, exists := schemaMap["properties"]; exists {
		if propsMap, ok := properties.(map[string]interface{}); ok {
			geminiSchema.Properties = make(map[string]*genai.Schema)
			for key, value := range propsMap {
				geminiSchema.Properties[key] = convertJSONSchemaToGeminiSchema(value)
			}
		}
	}

	// Handle required fields
	if required, exists := schemaMap["required"]; exists {
		if reqSlice, ok := required.([]interface{}); ok {
			requiredStrings := make([]string, len(reqSlice))
			for i, v := range reqSlice {
				if str, ok := v.(string); ok {
					requiredStrings[i] = str
				}
			}
			geminiSchema.Required = requiredStrings
		}
	}

	return geminiSchema
}

// handleToolCalls processes tool calls from the response
func (p *Provider) handleToolCalls(candidates []*genai.Candidate) []types.ToolCall {
	var toolCalls []types.ToolCall

	for _, candidate := range candidates {
		if candidate.Content != nil {
			for _, part := range candidate.Content.Parts {
				// Check if this part contains a function call
				if part != nil && part.FunctionCall != nil {
					funcCall := part.FunctionCall

					// Convert function call arguments to JSON string
					argsJSON := ""
					if funcCall.Args != nil {
						if jsonBytes, err := json.Marshal(funcCall.Args); err == nil {
							argsJSON = string(jsonBytes)
						}
					}

					// Use the function call ID if available, otherwise generate one
					callID := funcCall.ID
					if callID == "" {
						callID = fmt.Sprintf("call_%s_%d", funcCall.Name, len(toolCalls))
					}

					toolCall := types.ToolCall{
						ID:   callID,
						Type: "function",
						Function: types.ToolCallFunction{
							Name:      funcCall.Name,
							Arguments: argsJSON,
						},
						Args: funcCall.Args,
					}
					toolCalls = append(toolCalls, toolCall)
				}
			}
		}
	}

	return toolCalls
}
