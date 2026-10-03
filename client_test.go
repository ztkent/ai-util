package aiutil

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestChat(t *testing.T) {
	var got wireRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if auth := r.Header.Get("Authorization"); auth != "Bearer test-key" {
			t.Errorf("unexpected auth header: %q", auth)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		fmt.Fprint(w, `{
			"id":"gen-1","model":"test/model",
			"choices":[{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}
		}`)
	}))
	defer srv.Close()

	c := New("test-key", WithBaseURL(srv.URL))
	resp, err := c.Chat(context.Background(), &Request{
		Model:    "test/model",
		Messages: []Message{User("hi")},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Message.Content != "hello" {
		t.Errorf("content = %q, want hello", resp.Message.Content)
	}
	if resp.Usage.TotalTokens != 4 {
		t.Errorf("total tokens = %d, want 4", resp.Usage.TotalTokens)
	}
	if got.Model != "test/model" || len(got.Messages) != 1 || got.Messages[0].Role != "user" {
		t.Errorf("unexpected wire request: %+v", got)
	}
}

func TestChatDefaultsModel(t *testing.T) {
	var got wireRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&got)
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	}))
	defer srv.Close()

	c := New("k", WithBaseURL(srv.URL), WithDefaultModel("default/model"))
	if _, err := c.Chat(context.Background(), &Request{Messages: []Message{User("hi")}}); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if got.Model != "default/model" {
		t.Errorf("model = %q, want default/model", got.Model)
	}
}

func TestChatZeroTemperatureIsSent(t *testing.T) {
	var raw map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&raw)
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	}))
	defer srv.Close()

	zero := 0.0
	c := New("k", WithBaseURL(srv.URL))
	if _, err := c.Chat(context.Background(), &Request{
		Model:       "m",
		Messages:    []Message{User("hi")},
		Temperature: &zero,
	}); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if v, ok := raw["temperature"]; !ok || v.(float64) != 0 {
		t.Errorf("temperature = %v (present=%v), want 0", v, ok)
	}
}

func TestChatAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprint(w, `{"error":{"code":"invalid_api_key","message":"bad key"}}`)
	}))
	defer srv.Close()

	c := New("k", WithBaseURL(srv.URL))
	_, err := c.Chat(context.Background(), &Request{Model: "m", Messages: []Message{User("hi")}})
	apiErr, ok := IsAPIError(err)
	if !ok {
		t.Fatalf("expected *APIError, got %T: %v", err, err)
	}
	if apiErr.StatusCode != 401 || apiErr.Code != "invalid_api_key" {
		t.Errorf("unexpected error: %+v", apiErr)
	}
	if apiErr.Retryable() {
		t.Error("401 should not be retryable")
	}
}

func TestChatRetriesOn429(t *testing.T) {
	attempts := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 3 {
			w.WriteHeader(http.StatusTooManyRequests)
			fmt.Fprint(w, `{"error":{"message":"slow down"}}`)
			return
		}
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	}))
	defer srv.Close()

	c := New("k", WithBaseURL(srv.URL), WithRetry(RetryPolicy{MaxAttempts: 5, BaseDelay: 1}))
	if _, err := c.Chat(context.Background(), &Request{Model: "m", Messages: []Message{User("hi")}}); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if attempts != 3 {
		t.Errorf("attempts = %d, want 3", attempts)
	}
}

func TestChatStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		write := func(s string) {
			fmt.Fprintf(w, "data: %s\n\n", s)
			flusher.Flush()
		}
		fmt.Fprint(w, ": OPENROUTER PROCESSING\n\n")
		write(`{"id":"gen-1","model":"m","choices":[{"delta":{"role":"assistant","content":"Hel"}}]}`)
		write(`{"id":"gen-1","model":"m","choices":[{"delta":{"content":"lo"}}]}`)
		write(`{"id":"gen-1","model":"m","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":"{\"city\":"}}]}}]}`)
		write(`{"id":"gen-1","model":"m","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]}}]}`)
		write(`{"id":"gen-1","model":"m","choices":[{"delta":{},"finish_reason":"tool_calls"}]}`)
		write(`{"id":"gen-1","model":"m","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	c := New("k", WithBaseURL(srv.URL))
	var text strings.Builder
	var calls []ToolCall
	resp, err := c.ChatStream(context.Background(), &Request{Model: "m", Messages: []Message{User("hi")}}, func(e Event) error {
		switch e.Type {
		case EventText:
			text.WriteString(e.Text)
		case EventToolCall:
			calls = append(calls, e.ToolCall)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	if text.String() != "Hello" {
		t.Errorf("text = %q, want Hello", text.String())
	}
	if len(calls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(calls))
	}
	if calls[0].Name != "get_weather" || calls[0].Arguments != `{"city":"Paris"}` {
		t.Errorf("unexpected tool call: %+v", calls[0])
	}
	if resp.Message.Content != "Hello" || len(resp.Message.ToolCalls) != 1 {
		t.Errorf("unexpected response message: %+v", resp.Message)
	}
	if resp.Usage.TotalTokens != 7 {
		t.Errorf("total tokens = %d, want 7", resp.Usage.TotalTokens)
	}
	if resp.FinishReason != "tool_calls" {
		t.Errorf("finish reason = %q, want tool_calls", resp.FinishReason)
	}
}

func TestBuildRequestValidation(t *testing.T) {
	c := New("k")
	if _, err := c.Chat(context.Background(), &Request{Messages: []Message{User("hi")}}); err == nil {
		t.Error("expected error for missing model")
	}
	if _, err := c.Chat(context.Background(), &Request{Model: "m"}); err == nil {
		t.Error("expected error for missing messages")
	}
}

func TestChatStreamJSONError(t *testing.T) {
	// OpenRouter can return a JSON error body (HTTP 200) instead of SSE.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"error":{"message":"upstream rate limited","code":429}}`)
	}))
	defer srv.Close()

	c := New("k", WithBaseURL(srv.URL), WithRetry(RetryPolicy{MaxAttempts: 1}))
	_, err := c.ChatStream(context.Background(), &Request{Model: "m", Messages: []Message{User("hi")}}, nil)
	apiErr, ok := IsAPIError(err)
	if !ok {
		t.Fatalf("expected *APIError, got %T: %v", err, err)
	}
	if apiErr.StatusCode != 429 || !apiErr.Retryable() {
		t.Errorf("unexpected error: %+v", apiErr)
	}
}

func TestChatStreamErrorAfterDataIsNotRetried(t *testing.T) {
	// A failure after data has been emitted must surface, not be retried or
	// masked, since replaying would duplicate output.
	attempts := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n")
		// OpenRouter can inject an error object mid-stream.
		fmt.Fprint(w, `{"error":{"message":"upstream failed","code":429}}`)
		w.(http.Flusher).Flush()
	}))
	defer srv.Close()

	c := New("k", WithBaseURL(srv.URL), WithRetry(RetryPolicy{MaxAttempts: 3, BaseDelay: 1}))
	_, err := c.ChatStream(context.Background(), &Request{Model: "m", Messages: []Message{User("hi")}}, nil)
	if err == nil {
		t.Fatal("expected an error")
	}
	if attempts != 1 {
		t.Errorf("attempts = %d, want 1 (no retry after data)", attempts)
	}
}
