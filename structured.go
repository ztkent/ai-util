package aiutil

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// Schema builds a response_format value for structured output. name labels the
// schema; schema is a JSON Schema object. Pass the result as
// Request.ResponseSchema.
func Schema(name string, schema map[string]any) map[string]any {
	return map[string]any{
		"name":   name,
		"strict": true,
		"schema": schema,
	}
}

// RunJSON runs the agent and unmarshals the final message into v. It asks the
// model for JSON conforming to schema and, if the reply is not valid JSON,
// retries with a corrective message. The schema is applied per call, so it
// never leaks into the agent's state or other runs.
//
// Retries default to 1; override with WithMaxAttempts. Other RunOptions are
// passed through to each attempt.
func (a *Agent) RunJSON(ctx context.Context, input string, schema map[string]any, v any, opts ...RunOption) (*Response, error) {
	cfg := runConfig{maxAttempts: 1}
	for _, opt := range opts {
		opt(&cfg)
	}
	if cfg.maxAttempts <= 0 {
		cfg.maxAttempts = 1
	}

	// Apply the schema to every attempt, preserving any caller options.
	attemptOpts := append([]RunOption{WithSchema(schema)}, opts...)

	var lastErr error
	for attempt := 0; attempt < cfg.maxAttempts; attempt++ {
		resp, err := a.Run(ctx, input, attemptOpts...)
		if err != nil {
			return nil, err
		}
		if err := unmarshalJSON(resp.Message.Content, v); err != nil {
			lastErr = err
			input = fmt.Sprintf("Your previous reply was not valid JSON (%v). Reply with only valid JSON matching the schema.", err)
			continue
		}
		return resp, nil
	}
	return nil, fmt.Errorf("structured output: %w", lastErr)
}

// unmarshalJSON decodes JSON from content, tolerating markdown code fences.
func unmarshalJSON(content string, v any) error {
	content = strings.TrimSpace(content)
	if strings.HasPrefix(content, "```") {
		content = strings.TrimPrefix(content, "```json")
		content = strings.TrimPrefix(content, "```")
		content = strings.TrimSuffix(strings.TrimSpace(content), "```")
		content = strings.TrimSpace(content)
	}
	if err := json.Unmarshal([]byte(content), v); err != nil {
		return fmt.Errorf("decode JSON: %w", err)
	}
	return nil
}
