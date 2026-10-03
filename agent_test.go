package aiutil

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

// toolLoopServer returns a tool call on the first request and a final answer
// on the second, letting tests exercise the agent loop.
func toolLoopServer(t *testing.T) *httptest.Server {
	t.Helper()
	turn := 0
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		turn++
		if turn == 1 {
			fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","tool_calls":[
				{"id":"call_1","type":"function","function":{"name":"add","arguments":"{\"a\":2,\"b\":3}"}}
			]},"finish_reason":"tool_calls"}]}`)
			return
		}
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"The answer is 5"},"finish_reason":"stop"}]}`)
	}))
}

func TestAgentToolLoop(t *testing.T) {
	srv := toolLoopServer(t)
	defer srv.Close()

	var gotArgs struct{ A, B int }
	agent := NewAgent(New("k", WithBaseURL(srv.URL)), "m",
		WithSystem("be helpful"),
		WithTools(Tool{
			Name:        "add",
			Description: "add two numbers",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{"a": map[string]any{"type": "integer"}, "b": map[string]any{"type": "integer"}},
			},
			Handler: func(ctx context.Context, args json.RawMessage) (string, error) {
				if err := json.Unmarshal(args, &gotArgs); err != nil {
					return "", err
				}
				return fmt.Sprintf("%d", gotArgs.A+gotArgs.B), nil
			},
		}),
	)

	resp, err := agent.Run(context.Background(), "what is 2+3?")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if resp.Message.Content != "The answer is 5" {
		t.Errorf("content = %q", resp.Message.Content)
	}
	if gotArgs.A != 2 || gotArgs.B != 3 {
		t.Errorf("handler got %+v, want {2 3}", gotArgs)
	}

	// history: user, assistant(tool call), tool result, assistant(final)
	hist := agent.History()
	if len(hist) != 4 {
		t.Fatalf("history length = %d, want 4: %+v", len(hist), hist)
	}
	if hist[2].Role != RoleTool || hist[2].Content != "5" || hist[2].ToolCallID != "call_1" {
		t.Errorf("unexpected tool result message: %+v", hist[2])
	}
}

func TestAgentUnknownTool(t *testing.T) {
	turn := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		turn++
		if turn == 1 {
			fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","tool_calls":[
				{"id":"c1","type":"function","function":{"name":"missing","arguments":"{}"}}
			]}}]}`)
			return
		}
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"done"}}]}`)
	}))
	defer srv.Close()

	agent := NewAgent(New("k", WithBaseURL(srv.URL)), "m")
	if _, err := agent.Run(context.Background(), "go"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	hist := agent.History()
	if hist[2].Content != `error: unknown tool "missing"` {
		t.Errorf("unexpected tool result: %q", hist[2].Content)
	}
}

func TestAgentMaxTurns(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Always request a tool call, never finishing.
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","tool_calls":[
			{"id":"c1","type":"function","function":{"name":"loop","arguments":"{}"}}
		]}}]}`)
	}))
	defer srv.Close()

	agent := NewAgent(New("k", WithBaseURL(srv.URL)), "m",
		WithMaxTurns(3),
		WithTools(Tool{Name: "loop", Handler: func(context.Context, json.RawMessage) (string, error) { return "again", nil }}),
	)
	if _, err := agent.Run(context.Background(), "go"); err == nil {
		t.Fatal("expected max-turns error")
	}
}

func TestAgentRunStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	agent := NewAgent(New("k", WithBaseURL(srv.URL)), "m")
	var events []EventType
	resp, err := agent.RunStream(context.Background(), "hello", func(e Event) error {
		events = append(events, e.Type)
		return nil
	})
	if err != nil {
		t.Fatalf("RunStream: %v", err)
	}
	if resp.Message.Content != "hi" {
		t.Errorf("content = %q", resp.Message.Content)
	}
	if len(events) == 0 || events[len(events)-1] != EventDone {
		t.Errorf("expected final EventDone, got %v", events)
	}
}
