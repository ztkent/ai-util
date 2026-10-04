package aiutil

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
)

// ErrMaxTurns is returned when an agent exceeds MaxTurns without producing a
// final answer. Use errors.Is to detect it.
var ErrMaxTurns = errors.New("agent: exceeded max turns without a final answer")

// Agent runs a multi-turn conversation, executing tools and feeding their
// results back to the model until it produces a final answer.
//
// An Agent is safe for concurrent use, but a single Run holds the agent's lock
// for its duration. Callbacks (OnEvent, OnUsage, OnTurn) must not call back
// into the same Agent, or they will deadlock.
type Agent struct {
	client  ChatClient
	model   string
	system  string
	tools   []Tool
	history []Message

	// MaxTurns caps model calls per Run to prevent infinite tool loops.
	// Defaults to 10.
	MaxTurns int

	// MaxToolOutput truncates each tool result to this many bytes before it is
	// fed back to the model. 0 disables truncation.
	MaxToolOutput int

	// MaxContextTokens, when > 0, trims the oldest history before each model
	// call so the estimated prompt stays within the limit.
	MaxContextTokens int

	// OnEvent, when set, receives streaming events during RunStream.
	OnEvent func(Event) error

	// OnUsage, when set, is called after every model call with that call's
	// usage. Returning an error aborts the run.
	OnUsage func(Usage) error

	// OnTurn, when set, is called after every model call with the full
	// response. Returning an error aborts the run.
	OnTurn func(*Response) error

	mu sync.Mutex
}

// runConfig holds per-call options. It is built fresh for each Run, so options
// never leak between calls or across goroutines.
type runConfig struct {
	onEvent     func(Event) error
	schema      map[string]any
	jsonMode    bool
	maxAttempts int
}

// RunOption configures a single Run.
type RunOption func(*runConfig)

// WithSchema constrains this run's output to a JSON schema (see Schema).
func WithSchema(schema map[string]any) RunOption {
	return func(c *runConfig) { c.schema = schema }
}

// WithJSONMode asks this run for a JSON object without a schema.
func WithJSONMode() RunOption {
	return func(c *runConfig) { c.jsonMode = true }
}

// WithMaxAttempts sets how many times RunJSON retries invalid JSON.
func WithMaxAttempts(n int) RunOption {
	return func(c *runConfig) { c.maxAttempts = n }
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

// WithMaxToolOutput truncates each tool result to n bytes (0 disables).
func WithMaxToolOutput(n int) AgentOption { return func(a *Agent) { a.MaxToolOutput = n } }

// WithMaxContextTokens trims history to stay within an estimated token budget.
func WithMaxContextTokens(n int) AgentOption { return func(a *Agent) { a.MaxContextTokens = n } }

// WithOnUsage registers a callback invoked after every model call.
func WithOnUsage(fn func(Usage) error) AgentOption { return func(a *Agent) { a.OnUsage = fn } }

// WithOnTurn registers a callback invoked after every model call.
func WithOnTurn(fn func(*Response) error) AgentOption { return func(a *Agent) { a.OnTurn = fn } }

// WithHistory seeds the conversation with prior messages.
func WithHistory(msgs ...Message) AgentOption {
	return func(a *Agent) { a.history = append(a.history, msgs...) }
}

// NewAgent creates an Agent bound to a client and model. The client may be a
// *Client or any ChatClient implementation (for example, a cost-ledger wrapper).
func NewAgent(client ChatClient, model string, opts ...AgentOption) *Agent {
	a := &Agent{client: client, model: model, MaxTurns: 10}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// History returns a copy of the conversation so far.
func (a *Agent) History() []Message {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]Message, len(a.history))
	copy(out, a.history)
	return out
}

// Reset clears the conversation history.
func (a *Agent) Reset() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.history = nil
}

// Run sends a user message and drives the tool loop to completion, returning
// the model's final response. The full exchange is appended to history.
func (a *Agent) Run(ctx context.Context, input string, opts ...RunOption) (*Response, error) {
	return a.run(ctx, input, nil, opts...)
}

// RunStream is Run with streaming events delivered to onEvent (falling back to
// the agent's OnEvent when onEvent is nil).
func (a *Agent) RunStream(ctx context.Context, input string, onEvent func(Event) error, opts ...RunOption) (*Response, error) {
	if onEvent == nil {
		onEvent = a.OnEvent
	}
	return a.run(ctx, input, onEvent, opts...)
}

func (a *Agent) run(ctx context.Context, input string, onEvent func(Event) error, opts ...RunOption) (*Response, error) {
	cfg := runConfig{onEvent: onEvent}
	for _, opt := range opts {
		opt(&cfg)
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	if input != "" {
		a.history = append(a.history, User(input))
	}

	maxTurns := a.MaxTurns
	if maxTurns <= 0 {
		maxTurns = 10
	}

	for turn := 0; turn < maxTurns; turn++ {
		a.trimHistory()

		req := &Request{
			Model:          a.model,
			Messages:       a.messages(),
			Tools:          a.tools,
			ResponseSchema: cfg.schema,
			JSONMode:       cfg.jsonMode,
		}

		var (
			resp *Response
			err  error
		)
		if cfg.onEvent != nil {
			resp, err = a.client.ChatStream(ctx, req, cfg.onEvent)
		} else {
			resp, err = a.client.Chat(ctx, req)
		}
		if err != nil {
			return nil, err
		}

		a.history = append(a.history, resp.Message)

		if a.OnUsage != nil {
			if err := a.OnUsage(resp.Usage); err != nil {
				return resp, err
			}
		}
		if a.OnTurn != nil {
			if err := a.OnTurn(resp); err != nil {
				return resp, err
			}
		}

		if len(resp.Message.ToolCalls) == 0 {
			return resp, nil
		}

		for _, call := range resp.Message.ToolCalls {
			result := a.execute(ctx, call)
			a.history = append(a.history, ToolResult(call.ID, result))
			if cfg.onEvent != nil {
				if err := cfg.onEvent(Event{Type: EventToolResult, ToolCall: call, Result: result}); err != nil {
					return resp, err
				}
			}
		}
	}

	return nil, fmt.Errorf("%w (%d)", ErrMaxTurns, maxTurns)
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
		if t.Timeout > 0 {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, t.Timeout)
			defer cancel()
		}
		out, err := t.Handler(ctx, json.RawMessage(call.Arguments))
		if err != nil {
			return fmt.Sprintf("error: %v", err)
		}
		return a.truncate(out)
	}
	return fmt.Sprintf("error: unknown tool %q", call.Name)
}

// truncate caps tool output to MaxToolOutput bytes, appending a marker so the
// model knows the result was cut.
func (a *Agent) truncate(out string) string {
	if a.MaxToolOutput <= 0 || len(out) <= a.MaxToolOutput {
		return out
	}
	return out[:a.MaxToolOutput] + fmt.Sprintf("\n... [truncated %d bytes]", len(out)-a.MaxToolOutput)
}

// trimHistory drops the oldest messages until the estimated prompt fits within
// MaxContextTokens. It never leaves a tool result at the front of history.
func (a *Agent) trimHistory() {
	if a.MaxContextTokens <= 0 {
		return
	}
	budget := a.MaxContextTokens - estimateTokens(a.system) - 512
	if budget < 0 {
		budget = 0
	}
	for estimateHistory(a.history) > budget && len(a.history) > 1 {
		a.history = a.history[1:]
		for len(a.history) > 0 && a.history[0].Role == RoleTool {
			a.history = a.history[1:]
		}
	}
}

// estimateTokens approximates a token count as one token per four bytes.
func estimateTokens(s string) int { return len(s) / 4 }

func estimateHistory(msgs []Message) int {
	total := 0
	for _, m := range msgs {
		total += estimateTokens(m.Content) + 4
		for _, tc := range m.ToolCalls {
			total += estimateTokens(tc.Arguments) + 4
		}
	}
	return total
}
