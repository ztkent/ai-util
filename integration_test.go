//go:build integration

package aiutil

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

// Run with: OPENROUTER_API_KEY=... go test -tags=integration -v
//
// These tests hit the live OpenRouter API using free models. Override the
// model with OPENROUTER_TEST_MODEL if the default is unavailable.

func testClient(t *testing.T) *Client {
	t.Helper()
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY not set")
	}
	return New("")
}

func testModel() string {
	if m := os.Getenv("OPENROUTER_TEST_MODEL"); m != "" {
		return m
	}
	return "nvidia/nemotron-3-super-120b-a12b:free"
}

func TestIntegrationChat(t *testing.T) {
	c := testClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	resp, err := c.Chat(ctx, &Request{
		Model:     testModel(),
		Messages:  []Message{User("Reply with exactly the word: pong")},
		MaxTokens: 1500,
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if strings.TrimSpace(resp.Message.Content) == "" {
		t.Error("expected non-empty content")
	}
	t.Logf("model=%s content=%q usage=%+v", resp.Model, resp.Message.Content, resp.Usage)
}

func TestIntegrationStream(t *testing.T) {
	c := testClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var text strings.Builder
	resp, err := c.ChatStream(ctx, &Request{
		Model:     testModel(),
		Messages:  []Message{User("Count from 1 to 5, separated by spaces.")},
		MaxTokens: 1500,
	}, func(e Event) error {
		if e.Type == EventText {
			text.WriteString(e.Text)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	if text.Len() == 0 {
		t.Error("expected streamed text")
	}
	if resp.Message.Content != text.String() {
		t.Errorf("final content %q != streamed %q", resp.Message.Content, text.String())
	}
	t.Logf("streamed=%q usage=%+v", text.String(), resp.Usage)
}

func TestIntegrationToolLoop(t *testing.T) {
	c := testClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	var called bool
	agent := NewAgent(c, testModel(),
		WithSystem("You are a calculator. Always use the add tool for arithmetic."),
		WithTools(Tool{
			Name:        "add",
			Description: "Add two integers and return the sum.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"a": map[string]any{"type": "integer", "description": "first number"},
					"b": map[string]any{"type": "integer", "description": "second number"},
				},
				"required": []string{"a", "b"},
			},
			Handler: func(ctx context.Context, args json.RawMessage) (string, error) {
				var in struct{ A, B int }
				if err := json.Unmarshal(args, &in); err != nil {
					return "", err
				}
				called = true
				return fmt.Sprintf("%d", in.A+in.B), nil
			},
		}),
	)

	resp, err := agent.Run(ctx, "What is 17 + 25? Use the tool.")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !called {
		t.Error("expected the add tool to be called")
	}
	if !strings.Contains(resp.Message.Content, "42") {
		t.Errorf("expected answer to contain 42, got %q", resp.Message.Content)
	}
	t.Logf("final=%q turns=%d", resp.Message.Content, len(agent.History()))
}

func TestIntegrationJSONMode(t *testing.T) {
	c := testClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	resp, err := c.Chat(ctx, &Request{
		Model:     testModel(),
		Messages:  []Message{User(`Return a JSON object with a single key "ok" set to true.`)},
		JSONMode:  true,
		MaxTokens: 1500,
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	var out map[string]any
	if err := json.Unmarshal([]byte(resp.Message.Content), &out); err != nil {
		t.Fatalf("response is not valid JSON: %q (%v)", resp.Message.Content, err)
	}
	t.Logf("json=%v", out)
}
