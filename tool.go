package aiutil

import (
	"context"
	"encoding/json"
	"time"
)

// Tool is a function the model may call. Parameters is a JSON Schema object
// (typically a map[string]any) describing the arguments.
type Tool struct {
	Name        string
	Description string
	Parameters  any

	// Timeout, when > 0, bounds how long the handler may run.
	Timeout time.Duration

	// Handler runs the tool. The returned string is fed back to the model.
	// Returning an error feeds the error text back so the model can recover.
	Handler func(ctx context.Context, args json.RawMessage) (string, error)
}

// schema returns the tool's parameters as a JSON Schema object, defaulting to
// an empty object schema when none is provided.
func (t Tool) schema() map[string]any {
	if t.Parameters == nil {
		return map[string]any{"type": "object", "properties": map[string]any{}}
	}
	if m, ok := t.Parameters.(map[string]any); ok {
		return m
	}
	// Marshal/unmarshal to normalize structs and other map types.
	raw, err := json.Marshal(t.Parameters)
	if err != nil {
		return map[string]any{"type": "object", "properties": map[string]any{}}
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return map[string]any{"type": "object", "properties": map[string]any{}}
	}
	return m
}
