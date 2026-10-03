package aiutil

import (
	"context"
	"encoding/json"
	"fmt"
)

// Agent runs a multi-turn conversation, executing tools and feeding their
// results back to the model until it produces a final answer.
type Agent struct {
	client  *Client
	model   string
	system  string
	tools   []Tool
	history []Message

	// MaxTurns caps model calls per Run to prevent infinite tool loops.
	// Defaults to 10.
	MaxTurns int

	// OnEvent, when set, receives streaming events during RunStream.
	OnEvent func(Event) error
}

// AgentOption configures an Agent.
type AgentOption func(*Agent)

// WithSystem sets the system prompt.
func WithSystem(prompt string) AgentOption { return func(a *Agent) { a.system = prompt } }

// WithTools registers tools the agent may call.
func WithTools(tools ...Tool) AgentOption {
	return func(a *Agent) { a.tools = append(a.tools, tools...) }
}

// WithMaxTurns caps the number of model calls per Run.
func WithMaxTurns(n int) AgentOption { return func(a *Agent) { a.MaxTurns = n } }

// WithHistory seeds the conversation with prior messages.
func WithHistory(msgs ...Message) AgentOption {
	return func(a *Agent) { a.history = append(a.history, msgs...) }
}

// NewAgent creates an Agent bound to a client and model.
func NewAgent(client *Client, model string, opts ...AgentOption) *Agent {
	a := &Agent{client: client, model: model, MaxTurns: 10}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// History returns a copy of the conversation so far.
func (a *Agent) History() []Message {
	out := make([]Message, len(a.history))
	copy(out, a.history)
	return out
}

// Reset clears the conversation history.
func (a *Agent) Reset() { a.history = nil }

// Run sends a user message and drives the tool loop to completion, returning
// the model's final response. The full exchange is appended to history.
func (a *Agent) Run(ctx context.Context, input string) (*Response, error) {
	return a.run(ctx, input, nil)
}

// RunStream is Run with streaming events delivered to onEvent (falling back to
// the agent's OnEvent when onEvent is nil).
func (a *Agent) RunStream(ctx context.Context, input string, onEvent func(Event) error) (*Response, error) {
	if onEvent == nil {
		onEvent = a.OnEvent
	}
	return a.run(ctx, input, onEvent)
}

func (a *Agent) run(ctx context.Context, input string, onEvent func(Event) error) (*Response, error) {
	if input != "" {
		a.history = append(a.history, User(input))
	}

	maxTurns := a.MaxTurns
	if maxTurns <= 0 {
		maxTurns = 10
	}

	for turn := 0; turn < maxTurns; turn++ {
		req := &Request{
			Model:    a.model,
			Messages: a.messages(),
			Tools:    a.tools,
		}

		var (
			resp *Response
			err  error
		)
		if onEvent != nil {
			resp, err = a.client.ChatStream(ctx, req, onEvent)
		} else {
			resp, err = a.client.Chat(ctx, req)
		}
		if err != nil {
			return nil, err
		}

		a.history = append(a.history, resp.Message)

		if len(resp.Message.ToolCalls) == 0 {
			return resp, nil
		}

		for _, call := range resp.Message.ToolCalls {
			result := a.execute(ctx, call)
			a.history = append(a.history, ToolResult(call.ID, result))
			if onEvent != nil {
				if err := onEvent(Event{Type: EventToolResult, ToolCall: call, Result: result}); err != nil {
					return resp, err
				}
			}
		}
	}

	return nil, fmt.Errorf("agent: exceeded %d turns without a final answer", maxTurns)
}

// messages returns the system prompt (if any) followed by the history.
func (a *Agent) messages() []Message {
	if a.system == "" {
		return a.history
	}
	msgs := make([]Message, 0, len(a.history)+1)
	msgs = append(msgs, System(a.system))
	msgs = append(msgs, a.history...)
	return msgs
}

// execute runs a tool call, returning its output or an error string. Unknown
// tools and handler errors are reported back to the model rather than aborting.
func (a *Agent) execute(ctx context.Context, call ToolCall) string {
	for _, t := range a.tools {
		if t.Name != call.Name {
			continue
		}
		if t.Handler == nil {
			return fmt.Sprintf("error: tool %q has no handler", call.Name)
		}
		out, err := t.Handler(ctx, json.RawMessage(call.Arguments))
		if err != nil {
			return fmt.Sprintf("error: %v", err)
		}
		return out
	}
	return fmt.Sprintf("error: unknown tool %q", call.Name)
}
