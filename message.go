package aiutil

import "encoding/json"

// Role identifies who produced a message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ContentPart is one part of a multimodal message. Type is "text" or
// "image_url". For "text", Text is set; for "image_url", ImageURL is set.
type ContentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
}

// Message is a single entry in a conversation. It is a value type so that
// slices of messages are cheap to copy and never nil.
//
// Message is JSON-serializable so conversations can be persisted and restored
// (for example, to replay an agent run). Seed an agent with WithHistory.
type Message struct {
	Role    Role   `json:"role"`
	Content string `json:"content,omitempty"`

	// Parts, when set, carries multimodal content and takes precedence over
	// Content when the message is sent to the model.
	Parts []ContentPart `json:"parts,omitempty"`

	// ToolCalls is set on assistant messages that request tool execution.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`

	// ToolCallID links a RoleTool message back to the call it answers.
	ToolCallID string `json:"tool_call_id,omitempty"`

	// Name optionally identifies the tool or participant.
	Name string `json:"name,omitempty"`
}

// ToolCall is a model's request to run a tool.
type ToolCall struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // raw JSON arguments
}

// ParseArgs unmarshals the call's JSON arguments into v.
func (c ToolCall) ParseArgs(v any) error {
	if c.Arguments == "" {
		return nil
	}
	return json.Unmarshal([]byte(c.Arguments), v)
}

// System returns a system message.
func System(text string) Message { return Message{Role: RoleSystem, Content: text} }

// User returns a user message.
func User(text string) Message { return Message{Role: RoleUser, Content: text} }

// Assistant returns an assistant message.
func Assistant(text string) Message { return Message{Role: RoleAssistant, Content: text} }

// ToolResult returns a tool message carrying the output of a tool call.
func ToolResult(callID, content string) Message {
	return Message{Role: RoleTool, ToolCallID: callID, Content: content}
}

// UserParts returns a user message built from multimodal parts.
func UserParts(parts ...ContentPart) Message {
	return Message{Role: RoleUser, Parts: parts}
}

// TextPart returns a text content part.
func TextPart(text string) ContentPart { return ContentPart{Type: "text", Text: text} }

// ImagePart returns an image content part for the given URL (http(s) or data URI).
func ImagePart(url string) ContentPart { return ContentPart{Type: "image_url", ImageURL: url} }
