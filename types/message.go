package types

import (
	"time"
)

// Role represents the role of a message in a conversation
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// MessageContent represents different types of content that can be in a message
type MessageContent interface {
	Type() string
}

// TextContent represents text content
type TextContent struct {
	Text string `json:"text"`
}

func (t TextContent) Type() string { return "text" }

// ImageContent represents image content
type ImageContent struct {
	URL    string `json:"url,omitempty"`
	Base64 string `json:"base64,omitempty"`
	Detail string `json:"detail,omitempty"` // "low", "high", "auto"
}

func (i ImageContent) Type() string { return "image" }

// ToolCall represents a tool/function call
type ToolCall struct {
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Function ToolCallFunction       `json:"function"`
	Args     map[string]interface{} `json:"args,omitempty"`
}

type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolResult represents the result of a tool call
type ToolResult struct {
	ToolCallID string `json:"tool_call_id"`
	Content    string `json:"content"`
	Error      string `json:"error,omitempty"`
}

// Message represents a unified message format across all providers
type Message struct {
	ID         string                 `json:"id,omitempty"`
	Role       Role                   `json:"role"`
	Content    []MessageContent       `json:"content,omitempty"`
	TextData   string                 `json:"text_data,omitempty"` // For simple text messages
	ToolCalls  []ToolCall             `json:"tool_calls,omitempty"`
	ToolResult *ToolResult            `json:"tool_result,omitempty"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
	Timestamp  time.Time              `json:"timestamp,omitempty"`
}

// NewTextMessage creates a new text message
func NewTextMessage(role Role, text string) *Message {
	return &Message{
		Role:      role,
		TextData:  text,
		Timestamp: time.Now(),
	}
}

// NewContentMessage creates a message with structured content
func NewContentMessage(role Role, content []MessageContent) *Message {
	return &Message{
		Role:      role,
		Content:   content,
		Timestamp: time.Now(),
	}
}

// GetText returns the text content of the message
func (m *Message) GetText() string {
	if m.TextData != "" {
		return m.TextData
	}

	for _, content := range m.Content {
		if text, ok := content.(TextContent); ok {
			return text.Text
		}
	}

	return ""
}

// HasImages returns true if the message contains image content
func (m *Message) HasImages() bool {
	for _, content := range m.Content {
		if _, ok := content.(ImageContent); ok {
			return true
		}
	}
	return false
}
