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

// Message is a single entry in a conversation. It is a value type so that
// slices of messages are cheap to copy and never nil.
type Message struct {
	Role    Role
	Content string

	// ToolCalls is set on assistant messages that request tool execution.
	ToolCalls []ToolCall

	// ToolCallID links a RoleTool message back to the call it answers.
	ToolCallID string

	// Name optionally identifies the tool or participant.
	Name string
}

// ToolCall is a model's request to run a tool.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string // raw JSON arguments
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
