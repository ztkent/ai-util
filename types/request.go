package types

import (
	"context"
	"fmt"
)

// Error represents a structured error with provider context
type Error struct {
	Code     string                 `json:"code"`
	Message  string                 `json:"message"`
	Provider string                 `json:"provider"`
	Details  map[string]interface{} `json:"details,omitempty"`
	Cause    error                  `json:"-"`
}

func (e *Error) Error() string {
	if e.Provider != "" {
		return fmt.Sprintf("[%s] %s: %s", e.Provider, e.Code, e.Message)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

func (e *Error) Unwrap() error {
	return e.Cause
}

// Common error codes
const (
	ErrCodeInvalidConfig      = "INVALID_CONFIG"
	ErrCodeAuthentication     = "AUTHENTICATION_FAILED"
	ErrCodeRateLimit          = "RATE_LIMIT_EXCEEDED"
	ErrCodeQuotaExceeded      = "QUOTA_EXCEEDED"
	ErrCodeModelNotFound      = "MODEL_NOT_FOUND"
	ErrCodeInvalidRequest     = "INVALID_REQUEST"
	ErrCodeServerError        = "SERVER_ERROR"
	ErrCodeTimeout            = "TIMEOUT"
	ErrCodeTokenLimitExceeded = "TOKEN_LIMIT_EXCEEDED"
	ErrCodeContentFiltered    = "CONTENT_FILTERED"
)

// NewError creates a new structured error
func NewError(code, message, provider string) *Error {
	return &Error{
		Code:     code,
		Message:  message,
		Provider: provider,
		Details:  make(map[string]interface{}),
	}
}

// WrapError wraps an existing error with provider context
func WrapError(err error, code, provider string) *Error {
	return &Error{
		Code:     code,
		Message:  err.Error(),
		Provider: provider,
		Cause:    err,
		Details:  make(map[string]interface{}),
	}
}

// CompletionRequest represents a unified completion request
type CompletionRequest struct {
	Messages       []*Message             `json:"messages"`
	Model          string                 `json:"model"`
	MaxTokens      int                    `json:"max_tokens,omitempty"`
	Temperature    float64                `json:"temperature,omitempty"`
	TopP           float64                `json:"top_p,omitempty"`
	TopK           int                    `json:"top_k,omitempty"`
	Seed           *int                   `json:"seed,omitempty"`
	Stop           []string               `json:"stop,omitempty"`
	Stream         bool                   `json:"stream,omitempty"`
	Tools          []Tool                 `json:"tools,omitempty"`
	GroundingTools []GroundingTool        `json:"grounding_tools,omitempty"` // Google-specific: URL context, Google Search
	ToolChoice     interface{}            `json:"tool_choice,omitempty"`
	ThinkingConfig *ThinkingConfig        `json:"thinking_config,omitempty"`
	ResponseFormat *ResponseFormat        `json:"response_format,omitempty"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
}

// CompletionResponse represents a unified completion response
type CompletionResponse struct {
	ID           string                 `json:"id"`
	Model        string                 `json:"model"`
	Provider     string                 `json:"provider"`
	Message      *Message               `json:"message,omitempty"`
	FinishReason string                 `json:"finish_reason,omitempty"`
	Usage        *Usage                 `json:"usage,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
	Created      int64                  `json:"created,omitempty"`
}

// StreamResponse represents a streaming response chunk
type StreamResponse struct {
	ID           string                 `json:"id"`
	Model        string                 `json:"model"`
	Provider     string                 `json:"provider"`
	Delta        *Message               `json:"delta,omitempty"`
	FinishReason string                 `json:"finish_reason,omitempty"`
	Usage        *Usage                 `json:"usage,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// Usage represents token usage information
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Tool represents a function/tool that can be called by the model
type Tool struct {
	Type     string        `json:"type"`
	Function *ToolFunction `json:"function,omitempty"`
}

// GroundingTool represents Google-specific grounding tools (URL context, Google Search)
type GroundingTool struct {
	Type string `json:"type"` // "url_context" or "google_search"
}

type ThinkingConfig struct {
	// Optional. Indicates whether to include thoughts in the response. If true, thoughts
	// are returned only if the model supports thought and thoughts are available.
	IncludeThoughts bool `json:"includeThoughts,omitempty"`
	// Optional. Indicates the thinking budget in tokens.
	ThinkingBudget *int32 `json:"thinkingBudget,omitempty"`
}

// GroundingToolURLContext enables URL context retrieval
const GroundingToolURLContext = "url_context"

// GroundingToolGoogleSearch enables Google Search grounding
const GroundingToolGoogleSearch = "google_search"

type ToolFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

// ResponseFormat represents the format of the response
type ResponseFormat struct {
	Type   string                 `json:"type"` // "text" or "json_object"
	Schema map[string]interface{} `json:"schema,omitempty"`
}

// StreamCallback defines the signature for streaming callbacks
type StreamCallback func(ctx context.Context, response *StreamResponse) error
