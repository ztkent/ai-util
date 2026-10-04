package aiutil

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// EventType classifies a streaming event.
type EventType string

const (
	EventText       EventType = "text"        // a chunk of assistant text
	EventToolCall   EventType = "tool_call"   // a completed tool call
	EventToolResult EventType = "tool_result" // a tool's output
	EventUsage      EventType = "usage"       // token usage for the call
	EventDone       EventType = "done"        // stream finished; Response is set
)

// Event is emitted during streaming.
type Event struct {
	Type     EventType
	Text     string    // set for EventText
	ToolCall ToolCall  // set for EventToolCall
	Result   string    // set for EventToolResult
	Usage    Usage     // set for EventUsage
	Response *Response // set for EventDone
}

// ChatStream performs a streaming completion. onEvent is called for each text
// chunk and completed tool call; the final Response is returned and also
// delivered as an EventDone.
func (c *Client) ChatStream(ctx context.Context, req *Request, onEvent func(Event) error) (*Response, error) {
	wire, err := c.buildRequest(req, true)
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(wire)
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}

	var resp *Response
	emitted := false
	policy := c.retry
	if req.Retry != nil {
		policy = *req.Retry
	}
	err = policy.do(ctx, func() error {
		var callErr error
		resp, callErr = c.doStream(ctx, body, onEvent, &emitted)
		if callErr != nil && emitted {
			// Data was already delivered; replaying would duplicate output.
			return stopRetry(callErr)
		}
		return callErr
	})
	if err != nil {
		return nil, err
	}
	if resp == nil {
		return nil, fmt.Errorf("openrouter: stream ended without a response")
	}
	if onEvent != nil {
		if err := onEvent(Event{Type: EventDone, Response: resp}); err != nil {
			return resp, err
		}
	}
	return resp, nil
}

func (c *Client) doStream(ctx context.Context, body []byte, onEvent func(Event) error, emitted *bool) (*Response, error) {
	httpResp, err := c.post(ctx, body)
	if err != nil {
		return nil, err
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		raw, _ := io.ReadAll(httpResp.Body)
		return nil, parseAPIError(httpResp, raw)
	}

	return readStream(httpResp.Body, onEvent, emitted)
}

// readStream parses the SSE stream, accumulating text and tool calls.
func readStream(r io.Reader, onEvent func(Event) error, emitted *bool) (*Response, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var (
		resp      Response
		content   strings.Builder
		toolAccum = map[int]*toolCallAccum{}
		toolOrder []int
	)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			// OpenRouter may return a JSON error body instead of an SSE
			// stream; surface it rather than silently ending the stream.
			if trimmed := strings.TrimSpace(line); strings.HasPrefix(trimmed, "{") {
				var errBody struct {
					Error *wireError `json:"error"`
				}
				if json.Unmarshal([]byte(trimmed), &errBody) == nil && errBody.Error != nil {
					return nil, apiErrorFromWire(errBody.Error)
				}
			}
			continue // comments (": OPENROUTER PROCESSING") and blank lines
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			continue
		}

		var chunk wireChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return nil, fmt.Errorf("decode stream chunk: %w", err)
		}
		if chunk.Error != nil {
			return nil, &APIError{Message: chunk.Error.Message}
		}
		if chunk.ID != "" {
			resp.ID = chunk.ID
		}
		if chunk.Model != "" {
			resp.Model = chunk.Model
		}
		if chunk.Usage != nil {
			resp.Usage = usageFromWire(chunk.Usage)
			if onEvent != nil {
				if err := onEvent(Event{Type: EventUsage, Usage: resp.Usage}); err != nil {
					return nil, err
				}
			}
		}
		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]
		if choice.FinishReason != "" {
			resp.FinishReason = choice.FinishReason
		}

		if text, ok := choice.Delta.Content.(string); ok && text != "" {
			content.WriteString(text)
			*emitted = true
			if onEvent != nil {
				if err := onEvent(Event{Type: EventText, Text: text}); err != nil {
					return nil, err
				}
			}
		}

		for _, tc := range choice.Delta.ToolCalls {
			idx := 0
			if tc.Index != nil {
				idx = *tc.Index
			}
			acc, ok := toolAccum[idx]
			if !ok {
				acc = &toolCallAccum{}
				toolAccum[idx] = acc
				toolOrder = append(toolOrder, idx)
			}
			if tc.ID != "" {
				acc.id = tc.ID
			}
			if tc.Function.Name != "" {
				acc.name = tc.Function.Name
			}
			acc.args.WriteString(tc.Function.Arguments)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read stream: %w", err)
	}

	resp.Message = Message{Role: RoleAssistant, Content: content.String()}
	for _, idx := range toolOrder {
		acc := toolAccum[idx]
		call := ToolCall{ID: acc.id, Name: acc.name, Arguments: acc.args.String()}
		resp.Message.ToolCalls = append(resp.Message.ToolCalls, call)
		*emitted = true
		if onEvent != nil {
			if err := onEvent(Event{Type: EventToolCall, ToolCall: call}); err != nil {
				return nil, err
			}
		}
	}
	return &resp, nil
}

type toolCallAccum struct {
	id   string
	name string
	args strings.Builder
}
