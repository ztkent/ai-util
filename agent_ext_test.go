package aiutil

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// ledgerClient wraps a *Client to record usage, exercising the ChatClient
// interface and the per-turn usage hook.
type ledgerClient struct {
	inner *Client
	calls int
	total Usage
}

func (l *ledgerClient) Chat(ctx context.Context, req *Request) (*Response, error) {
	l.calls++
	resp, err := l.inner.Chat(ctx, req)
	if err == nil {
		l.total = l.total.Add(resp.Usage)
	}
	return resp, err
}

func (l *ledgerClient) ChatStream(ctx context.Context, req *Request, onEvent func(Event) error) (*Response, error) {
	l.calls++
	resp, err := l.inner.ChatStream(ctx, req, onEvent)
	if err == nil {
		l.total = l.total.Add(resp.Usage)
	}
	return resp, err
}

func TestAgentAcceptsChatClient(t *testing.T) {
	srv := toolLoopServer(t)
	defer srv.Close()

	ledger := &ledgerClient{inner: New("k", WithBaseURL(srv.URL))}
	agent := NewAgent(ledger, "m", WithTools(Tool{
		Name:    "add",
		Handler: func(context.Context, json.RawMessage) (string, error) { return "5", nil },
	}))

	if _, err := agent.Run(context.Background(), "2+3?"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if ledger.calls != 2 {
		t.Errorf("wrapper calls = %d, want 2", ledger.calls)
	}
}

func TestAgentOnUsageAndOnTurn(t *testing.T) {
	srv := toolLoopServer(t)
	defer srv.Close()

	var usageCalls, turnCalls int
	var total Usage
	agent := NewAgent(New("k", WithBaseURL(srv.URL)), "m",
		WithTools(Tool{Name: "add", Handler: func(context.Context, json.RawMessage) (string, error) { return "5", nil }}),
		WithOnUsage(func(u Usage) error { usageCalls++; total = total.Add(u); return nil }),
		WithOnTurn(func(*Response) error { turnCalls++; return nil }),
	)

	if _, err := agent.Run(context.Background(), "2+3?"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	// Two model calls: the tool-call turn and the final answer.
	if usageCalls != 2 || turnCalls != 2 {
		t.Errorf("usageCalls=%d turnCalls=%d, want 2 and 2", usageCalls, turnCalls)
	}
}

func TestAgentOnUsageAborts(t *testing.T) {
	srv := toolLoopServer(t)
	defer srv.Close()

	sentinel := errors.New("budget exceeded")
	agent := NewAgent(New("k", WithBaseURL(srv.URL)), "m",
		WithTools(Tool{Name: "add", Handler: func(context.Context, json.RawMessage) (string, error) { return "5", nil }}),
		WithOnUsage(func(Usage) error { return sentinel }),
	)

	_, err := agent.Run(context.Background(), "2+3?")
	if !errors.Is(err, sentinel) {
		t.Fatalf("err = %v, want sentinel", err)
	}
}

func TestAgentMaxTurnsSentinel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","tool_calls":[
			{"id":"c1","type":"function","function":{"name":"loop","arguments":"{}"}}
		]}}]}`)
	}))
	defer srv.Close()

	agent := NewAgent(New("k", WithBaseURL(srv.URL)), "m",
		WithMaxTurns(2),
		WithTools(Tool{Name: "loop", Handler: func(context.Context, json.RawMessage) (string, error) { return "again", nil }}),
	)
	_, err := agent.Run(context.Background(), "go")
	if !errors.Is(err, ErrMaxTurns) {
		t.Fatalf("err = %v, want ErrMaxTurns", err)
	}
}

func TestAgentToolTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","tool_calls":[
			{"id":"c1","type":"function","function":{"name":"slow","arguments":"{}"}}
		]}}]}`)
	}))
	defer srv.Close()

	agent := NewAgent(New("k", WithBaseURL(srv.URL)), "m",
		WithMaxTurns(1),
		WithTools(Tool{
			Name:    "slow",
			Timeout: 10 * time.Millisecond,
			Handler: func(ctx context.Context, _ json.RawMessage) (string, error) {
				select {
				case <-ctx.Done():
					return "", ctx.Err()
				case <-time.After(time.Second):
					return "done", nil
				}
			},
		}),
	)
	_, err := agent.Run(context.Background(), "go")
	if !errors.Is(err, ErrMaxTurns) {
		t.Fatalf("err = %v, want ErrMaxTurns", err)
	}
	hist := agent.History()
	if !strings.Contains(hist[2].Content, "context deadline exceeded") {
		t.Errorf("tool result = %q, want a timeout error", hist[2].Content)
	}
}

func TestAgentTruncatesToolOutput(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","tool_calls":[
			{"id":"c1","type":"function","function":{"name":"big","arguments":"{}"}}
		]}}]}`)
	}))
	defer srv.Close()

	agent := NewAgent(New("k", WithBaseURL(srv.URL)), "m",
		WithMaxTurns(1),
		WithMaxToolOutput(10),
		WithTools(Tool{Name: "big", Handler: func(context.Context, json.RawMessage) (string, error) {
			return strings.Repeat("x", 100), nil
		}}),
	)
	agent.Run(context.Background(), "go")
	hist := agent.History()
	if !strings.Contains(hist[2].Content, "truncated") {
		t.Errorf("tool result = %q, want truncation marker", hist[2].Content)
	}
}

func TestAgentHistoryJSONRoundTrip(t *testing.T) {
	msgs := []Message{
		System("be helpful"),
		User("hi"),
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Name: "add", Arguments: `{"a":1}`}}},
		ToolResult("c1", "1"),
	}
	raw, err := json.Marshal(msgs)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back []Message
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(back) != len(msgs) || back[2].ToolCalls[0].Name != "add" || back[3].ToolCallID != "c1" {
		t.Errorf("round trip mismatch: %+v", back)
	}
}

func TestAgentWithHistorySeedsConversation(t *testing.T) {
	var got wireRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&got)
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	}))
	defer srv.Close()

	agent := NewAgent(New("k", WithBaseURL(srv.URL)), "m",
		WithHistory(System("sys"), User("earlier")),
	)
	if _, err := agent.Run(context.Background(), "now"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(got.Messages) != 3 {
		t.Fatalf("messages = %d, want 3", len(got.Messages))
	}
	if got.Messages[0].Role != "system" || got.Messages[1].Content != "earlier" {
		t.Errorf("unexpected seeded history: %+v", got.Messages)
	}
}

func TestChatStreamEmitsUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
		w.(http.Flusher).Flush()
	}))
	defer srv.Close()

	var usageEvents int
	_, err := New("k", WithBaseURL(srv.URL)).ChatStream(context.Background(),
		&Request{Model: "m", Messages: []Message{User("hi")}},
		func(e Event) error {
			if e.Type == EventUsage {
				usageEvents++
				if e.Usage.TotalTokens != 3 {
					t.Errorf("usage = %+v, want 3 total", e.Usage)
				}
			}
			return nil
		})
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	if usageEvents != 1 {
		t.Errorf("usage events = %d, want 1", usageEvents)
	}
}

func TestChatPerRequestRetry(t *testing.T) {
	attempts := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 2 {
			w.WriteHeader(http.StatusTooManyRequests)
			fmt.Fprint(w, `{"error":{"message":"slow down"}}`)
			return
		}
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	}))
	defer srv.Close()

	// Client default is a single attempt; the request overrides it.
	c := New("k", WithBaseURL(srv.URL), WithRetry(RetryPolicy{MaxAttempts: 1}))
	policy := RetryPolicy{MaxAttempts: 3, BaseDelay: time.Millisecond}
	if _, err := c.Chat(context.Background(), &Request{
		Model:    "m",
		Messages: []Message{User("hi")},
		Retry:    &policy,
	}); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if attempts != 2 {
		t.Errorf("attempts = %d, want 2", attempts)
	}
}

func TestChatResponseSchema(t *testing.T) {
	var raw map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&raw)
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"{}"}}]}`)
	}))
	defer srv.Close()

	schema := Schema("answer", map[string]any{
		"type":       "object",
		"properties": map[string]any{"n": map[string]any{"type": "integer"}},
	})
	if _, err := New("k", WithBaseURL(srv.URL)).Chat(context.Background(), &Request{
		Model:          "m",
		Messages:       []Message{User("hi")},
		ResponseSchema: schema,
	}); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	rf, ok := raw["response_format"].(map[string]any)
	if !ok || rf["type"] != "json_schema" {
		t.Fatalf("response_format = %+v, want json_schema", raw["response_format"])
	}
}

func TestRunJSON(t *testing.T) {
	turn := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		turn++
		if turn == 1 {
			fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"not json"}}]}`)
			return
		}
		fenced := "```json\n{\"n\":7}\n```"
		body, _ := json.Marshal(map[string]any{
			"choices": []any{map[string]any{
				"message": map[string]any{"role": "assistant", "content": fenced},
			}},
		})
		w.Write(body)
	}))
	defer srv.Close()

	agent := NewAgent(New("k", WithBaseURL(srv.URL)), "m")
	var out struct{ N int }
	if _, err := agent.RunJSON(context.Background(), "give me n", nil, &out, WithMaxAttempts(3)); err != nil {
		t.Fatalf("RunJSON: %v", err)
	}
	if out.N != 7 {
		t.Errorf("n = %d, want 7", out.N)
	}
}

func TestRunJSONSchemaDoesNotLeak(t *testing.T) {
	var sawSchema []bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var raw map[string]any
		json.NewDecoder(r.Body).Decode(&raw)
		_, ok := raw["response_format"]
		sawSchema = append(sawSchema, ok)
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"{\"n\":1}"}}]}`)
	}))
	defer srv.Close()

	agent := NewAgent(New("k", WithBaseURL(srv.URL)), "m")
	var out struct{ N int }
	if _, err := agent.RunJSON(context.Background(), "give me n", Schema("x", map[string]any{"type": "object"}), &out); err != nil {
		t.Fatalf("RunJSON: %v", err)
	}
	// A plain Run afterwards must not carry the schema.
	if _, err := agent.Run(context.Background(), "plain"); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(sawSchema) != 2 || !sawSchema[0] || sawSchema[1] {
		t.Errorf("schema presence = %v, want [true false]", sawSchema)
	}
}

func TestRunWithSchemaOption(t *testing.T) {
	var raw map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&raw)
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"{}"}}]}`)
	}))
	defer srv.Close()

	agent := NewAgent(New("k", WithBaseURL(srv.URL)), "m")
	if _, err := agent.Run(context.Background(), "hi", WithSchema(Schema("x", map[string]any{"type": "object"}))); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if _, ok := raw["response_format"]; !ok {
		t.Error("expected response_format from WithSchema")
	}
}

func TestMultimodalMessage(t *testing.T) {
	var raw map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&raw)
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	}))
	defer srv.Close()

	msg := UserParts(TextPart("what is this?"), ImagePart("https://example.com/a.png"))
	if _, err := New("k", WithBaseURL(srv.URL)).Chat(context.Background(), &Request{
		Model:    "m",
		Messages: []Message{msg},
	}); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	msgs := raw["messages"].([]any)
	content := msgs[0].(map[string]any)["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("content parts = %d, want 2", len(content))
	}
	if content[1].(map[string]any)["type"] != "image_url" {
		t.Errorf("second part = %+v, want image_url", content[1])
	}
}

func TestEmbed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Errorf("path = %q, want /embeddings", r.URL.Path)
		}
		fmt.Fprint(w, `{"model":"emb","data":[{"index":0,"embedding":[0.1,0.2]}],"usage":{"total_tokens":2}}`)
	}))
	defer srv.Close()

	resp, err := New("k", WithBaseURL(srv.URL)).Embed(context.Background(), &EmbeddingRequest{
		Model: "emb",
		Input: []string{"hello"},
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(resp.Embeddings) != 1 || len(resp.Embeddings[0].Embedding) != 2 {
		t.Errorf("unexpected embeddings: %+v", resp.Embeddings)
	}
}

func TestListModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" {
			t.Errorf("path = %q, want /models", r.URL.Path)
		}
		fmt.Fprint(w, `{"data":[{"id":"a/b","name":"B","context_length":1000,"pricing":{"prompt":"0.000001","completion":"0.000002"}}]}`)
	}))
	defer srv.Close()

	models, err := New("k", WithBaseURL(srv.URL)).ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	if len(models) != 1 || models[0].ID != "a/b" || models[0].Pricing.Prompt != 0.000001 {
		t.Errorf("unexpected models: %+v", models)
	}
}
