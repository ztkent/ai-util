//go:build integration
// +build integration

package aiutil

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/ztkent/ai-util/types"
)

// These tests require actual API keys and should be run with:
// go test -tags=integration -v

func TestOpenAIIntegration(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set, skipping integration test")
	}

	// Create client with OpenAI provider
	client, err := NewAIClient().
		WithOpenAI(apiKey).
		WithDefaultProvider("openai").
		WithDefaultModel("gpt-4o-mini").
		Build()
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// Test basic completion
	ctx := context.Background()
	req := &types.CompletionRequest{
		Messages: []*types.Message{
			types.NewTextMessage(types.RoleUser, "Say hello"),
		},
		MaxTokens:   50,
		Temperature: 0.7,
	}

	resp, err := client.Complete(ctx, req)
	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}

	if resp.Message == nil || resp.Message.GetText() == "" {
		t.Error("Expected non-empty response")
	}

	t.Logf("OpenAI Response: %s", resp.Message.GetText())
}

func TestReplicateIntegration(t *testing.T) {
	apiKey := os.Getenv("REPLICATE_API_TOKEN")
	if apiKey == "" {
		t.Skip("REPLICATE_API_TOKEN not set, skipping integration test")
	}

	// Create client with Replicate provider
	client, err := NewAIClient().
		WithReplicate(apiKey).
		WithDefaultProvider("replicate").
		WithDefaultModel("meta/meta-llama-3-8b-instruct").
		Build()
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// Test basic completion
	ctx := context.Background()
	req := &types.CompletionRequest{
		Messages: []*types.Message{
			types.NewTextMessage(types.RoleUser, "Say hello"),
		},
		MaxTokens:   50,
		Temperature: 0.7,
	}

	resp, err := client.Complete(ctx, req)
	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}

	if resp.Message == nil || resp.Message.GetText() == "" {
		t.Error("Expected non-empty response")
	}

	t.Logf("Replicate Response: %s", resp.Message.GetText())
}

func TestGoogleIntegration(t *testing.T) {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		t.Skip("GOOGLE_API_KEY not set, skipping integration test")
	}

	// Create client with Google provider
	client, err := NewAIClient().
		WithGoogle(apiKey, "").
		WithDefaultProvider("google").
		WithDefaultModel("gemini-2.5-flash").
		Build()
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// Test basic completion
	ctx := context.Background()
	req := &types.CompletionRequest{
		Messages: []*types.Message{
			types.NewTextMessage(types.RoleUser, "Say hello"),
		},
		MaxTokens:   50,
		Temperature: 0.7,
	}

	resp, err := client.Complete(ctx, req)
	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}

	if resp.Message == nil || resp.Message.GetText() == "" {
		t.Error("Expected non-empty response")
	}

	t.Logf("Google Response: %s", resp.Message.GetText())
}

func TestConversationIntegration(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set, skipping integration test")
	}

	// Create client
	client, err := NewAIClient().
		WithOpenAI(apiKey).
		WithDefaultProvider("openai").
		WithDefaultModel("gpt-4o-mini").
		Build()
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// Create conversation
	conv := client.NewConversation(&ConversationConfig{
		SystemPrompt: "You are a helpful assistant. Keep responses brief.",
		MaxTokens:    1000,
	})

	// Test conversation flow
	ctx := context.Background()

	// Send message using conversation's Send method
	resp, err := conv.Send(ctx, "What is 2+2?", "gpt-4o-mini")
	if err != nil {
		t.Fatalf("Conversation send failed: %v", err)
	}

	if resp.Message == nil || resp.Message.GetText() == "" {
		t.Error("Expected non-empty response")
	}

	t.Logf("Conversation Response: %s", resp.Message.GetText())

	// Check that conversation now has 3 messages (system, user, assistant)
	messages := conv.GetMessages()
	if len(messages) != 3 {
		t.Errorf("Expected 3 messages, got %d", len(messages))
	}
}

// Streaming integration tests

func TestOpenAIStreamingIntegration(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set, skipping integration test")
	}

	// Create client with OpenAI provider
	client, err := NewAIClient().
		WithOpenAI(apiKey).
		WithDefaultProvider("openai").
		WithDefaultModel("gpt-4o-mini").
		Build()
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// Test streaming completion
	ctx := context.Background()
	req := &types.CompletionRequest{
		Messages: []*types.Message{
			types.NewTextMessage(types.RoleUser, "Count from 1 to 5, one number per line"),
		},
		MaxTokens:   100,
		Temperature: 0.1, // Low temperature for predictable output
		Stream:      true,
	}

	var streamedContent string
	var chunkCount int
	var lastUsage *types.Usage

	err = client.Stream(ctx, req, func(ctx context.Context, response *types.StreamResponse) error {
		chunkCount++

		if response.Delta != nil && response.Delta.TextData != "" {
			streamedContent += response.Delta.TextData
			t.Logf("OpenAI Stream Chunk %d: %q", chunkCount, response.Delta.TextData)
		}

		if response.Usage != nil {
			lastUsage = response.Usage
		}

		// Validate response structure
		if response.ID == "" {
			t.Error("Expected non-empty response ID")
		}
		if response.Model == "" {
			t.Error("Expected non-empty model name")
		}
		if response.Provider != "openai" {
			t.Errorf("Expected provider 'openai', got %s", response.Provider)
		}

		return nil
	})

	if err != nil {
		t.Fatalf("Streaming failed: %v", err)
	}

	// Validate streaming results
	if streamedContent == "" {
		t.Error("Expected non-empty streamed content")
	}
	if chunkCount == 0 {
		t.Error("Expected at least one streaming chunk")
	}

	t.Logf("OpenAI Streaming: Received %d chunks, total content: %q", chunkCount, streamedContent)
	t.Logf("OpenAI Usage: %+v", lastUsage)
}

func TestGoogleStreamingIntegration(t *testing.T) {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		t.Skip("GOOGLE_API_KEY not set, skipping integration test")
	}

	// Create client with Google provider
	client, err := NewAIClient().
		WithGoogle(apiKey, "").
		WithDefaultProvider("google").
		WithDefaultModel("gemini-2.5-flash").
		Build()
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// Test streaming completion
	ctx := context.Background()
	req := &types.CompletionRequest{
		Messages: []*types.Message{
			types.NewTextMessage(types.RoleUser, "Write a very short haiku about coding"),
		},
		MaxTokens:   100,
		Temperature: 0.3,
		Stream:      true,
	}

	var streamedContent string
	var chunkCount int
	var lastUsage *types.Usage

	err = client.Stream(ctx, req, func(ctx context.Context, response *types.StreamResponse) error {
		chunkCount++

		if response.Delta != nil && response.Delta.TextData != "" {
			streamedContent += response.Delta.TextData
			t.Logf("Google Stream Chunk %d: %q", chunkCount, response.Delta.TextData)
		}

		if response.Usage != nil {
			lastUsage = response.Usage
		}

		// Validate response structure
		if response.ID == "" {
			t.Error("Expected non-empty response ID")
		}
		if response.Model == "" {
			t.Error("Expected non-empty model name")
		}
		if response.Provider != "google" {
			t.Errorf("Expected provider 'google', got %s", response.Provider)
		}

		return nil
	})

	if err != nil {
		t.Fatalf("Google streaming failed: %v", err)
	}

	// Validate streaming results
	if streamedContent == "" {
		t.Error("Expected non-empty streamed content")
	}
	if chunkCount == 0 {
		t.Error("Expected at least one streaming chunk")
	}

	t.Logf("Google Streaming: Received %d chunks, total content: %q", chunkCount, streamedContent)
	if lastUsage != nil {
		t.Logf("Google Usage: %+v", lastUsage)
	}
}

func TestReplicateStreamingIntegration(t *testing.T) {
	apiKey := os.Getenv("REPLICATE_API_TOKEN")
	if apiKey == "" {
		t.Skip("REPLICATE_API_TOKEN not set, skipping integration test")
	}

	// Create client with Replicate provider
	client, err := NewAIClient().
		WithReplicate(apiKey).
		WithDefaultProvider("replicate").
		WithDefaultModel("meta/meta-llama-3-8b-instruct").
		Build()
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// Test streaming completion
	ctx := context.Background()
	req := &types.CompletionRequest{
		Messages: []*types.Message{
			types.NewTextMessage(types.RoleUser, "Say hello and explain what you are in one sentence"),
		},
		MaxTokens:   150,
		Temperature: 0.2,
		Stream:      true,
	}

	var streamedContent string
	var chunkCount int
	var lastUsage *types.Usage

	err = client.Stream(ctx, req, func(ctx context.Context, response *types.StreamResponse) error {
		chunkCount++

		if response.Delta != nil && response.Delta.TextData != "" {
			streamedContent += response.Delta.TextData
			t.Logf("Replicate Stream Chunk %d: %q", chunkCount, response.Delta.TextData)
		}

		if response.Usage != nil {
			lastUsage = response.Usage
		}

		// Validate response structure
		if response.ID == "" {
			t.Error("Expected non-empty response ID")
		}
		if response.Model == "" {
			t.Error("Expected non-empty model name")
		}
		if response.Provider != "replicate" {
			t.Errorf("Expected provider 'replicate', got %s", response.Provider)
		}

		return nil
	})

	if err != nil {
		t.Fatalf("Replicate streaming failed: %v", err)
	}

	// Validate streaming results
	if streamedContent == "" {
		t.Error("Expected non-empty streamed content")
	}
	if chunkCount == 0 {
		t.Error("Expected at least one streaming chunk")
	}

	t.Logf("Replicate Streaming: Received %d chunks, total content: %q", chunkCount, streamedContent)
	if lastUsage != nil {
		t.Logf("Replicate Usage: %+v", lastUsage)
	}
}

func TestConversationStreamingIntegration(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set, skipping integration test")
	}

	// Create client
	client, err := NewAIClient().
		WithOpenAI(apiKey).
		WithDefaultProvider("openai").
		WithDefaultModel("gpt-4o-mini").
		Build()
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// Create conversation
	conv := client.NewConversation(&ConversationConfig{
		SystemPrompt: "You are a helpful assistant. Keep responses very brief.",
		MaxTokens:    1000,
	})

	// Test conversation streaming
	ctx := context.Background()

	var streamedContent string
	var chunkCount int

	err = conv.SendStream(ctx, "What is 5 + 3?", "gpt-4o-mini", func(ctx context.Context, response *types.StreamResponse) error {
		chunkCount++

		if response.Delta != nil && response.Delta.TextData != "" {
			streamedContent += response.Delta.TextData
			t.Logf("Conversation Stream Chunk %d: %q", chunkCount, response.Delta.TextData)
		}

		return nil
	})

	if err != nil {
		t.Fatalf("Conversation streaming failed: %v", err)
	}

	// Validate streaming results
	if streamedContent == "" {
		t.Error("Expected non-empty streamed content")
	}
	if chunkCount == 0 {
		t.Error("Expected at least one streaming chunk")
	}

	// Check that conversation has been updated with both user and assistant messages
	messages := conv.GetMessages()
	expectedMessageCount := 3 // system + user + assistant
	if len(messages) != expectedMessageCount {
		t.Errorf("Expected %d messages after streaming, got %d", expectedMessageCount, len(messages))
	}

	// Check that the last message contains the streamed content
	if len(messages) > 0 {
		lastMessage := messages[len(messages)-1]
		if lastMessage.Role != types.RoleAssistant {
			t.Error("Expected last message to be from assistant")
		}
		if lastMessage.GetText() != streamedContent {
			t.Errorf("Expected last message content to match streamed content.\nExpected: %q\nGot: %q",
				streamedContent, lastMessage.GetText())
		}
	}

	t.Logf("Conversation Streaming: Received %d chunks, final content: %q", chunkCount, streamedContent)
}

func TestMultiProviderStreamingComparison(t *testing.T) {
	// Check which providers are available
	openaiKey := os.Getenv("OPENAI_API_KEY")
	googleKey := os.Getenv("GOOGLE_API_KEY")
	replicateKey := os.Getenv("REPLICATE_API_TOKEN")

	if openaiKey == "" && googleKey == "" && replicateKey == "" {
		t.Skip("No API keys set, skipping multi-provider streaming test")
	}

	prompt := "Explain what AI is in exactly one sentence."

	type ProviderResult struct {
		Name       string
		Content    string
		ChunkCount int
		Usage      *types.Usage
		Error      error
	}

	var results []ProviderResult
	ctx := context.Background()

	// Test OpenAI if available
	if openaiKey != "" {
		client, err := NewAIClient().
			WithOpenAI(openaiKey).
			WithDefaultProvider("openai").
			WithDefaultModel("gpt-4o-mini").
			Build()
		if err == nil {
			var content string
			var chunks int
			var usage *types.Usage

			req := &types.CompletionRequest{
				Messages: []*types.Message{
					types.NewTextMessage(types.RoleUser, prompt),
				},
				MaxTokens:   100,
				Temperature: 0.1,
				Stream:      true,
			}

			streamErr := client.Stream(ctx, req, func(ctx context.Context, response *types.StreamResponse) error {
				chunks++
				if response.Delta != nil && response.Delta.TextData != "" {
					content += response.Delta.TextData
				}
				if response.Usage != nil {
					usage = response.Usage
				}
				return nil
			})

			results = append(results, ProviderResult{
				Name:       "OpenAI",
				Content:    content,
				ChunkCount: chunks,
				Usage:      usage,
				Error:      streamErr,
			})

			client.Close()
		}
	}

	// Test Google if available
	if googleKey != "" {
		client, err := NewAIClient().
			WithGoogle(googleKey, "").
			WithDefaultProvider("google").
			WithDefaultModel("gemini-2.5-flash").
			Build()
		if err == nil {
			var content string
			var chunks int
			var usage *types.Usage

			req := &types.CompletionRequest{
				Messages: []*types.Message{
					types.NewTextMessage(types.RoleUser, prompt),
				},
				MaxTokens:   100,
				Temperature: 0.1,
				Stream:      true,
			}

			streamErr := client.Stream(ctx, req, func(ctx context.Context, response *types.StreamResponse) error {
				chunks++
				if response.Delta != nil && response.Delta.TextData != "" {
					content += response.Delta.TextData
				}
				if response.Usage != nil {
					usage = response.Usage
				}
				return nil
			})

			results = append(results, ProviderResult{
				Name:       "Google",
				Content:    content,
				ChunkCount: chunks,
				Usage:      usage,
				Error:      streamErr,
			})

			client.Close()
		}
	}

	// Test Replicate if available
	if replicateKey != "" {
		client, err := NewAIClient().
			WithReplicate(replicateKey).
			WithDefaultProvider("replicate").
			WithDefaultModel("meta/meta-llama-3-8b-instruct").
			Build()
		if err == nil {
			var content string
			var chunks int
			var usage *types.Usage

			req := &types.CompletionRequest{
				Messages: []*types.Message{
					types.NewTextMessage(types.RoleUser, prompt),
				},
				MaxTokens:   100,
				Temperature: 0.1,
				Stream:      true,
			}

			streamErr := client.Stream(ctx, req, func(ctx context.Context, response *types.StreamResponse) error {
				chunks++
				if response.Delta != nil && response.Delta.TextData != "" {
					content += response.Delta.TextData
				}
				if response.Usage != nil {
					usage = response.Usage
				}
				return nil
			})

			results = append(results, ProviderResult{
				Name:       "Replicate",
				Content:    content,
				ChunkCount: chunks,
				Usage:      usage,
				Error:      streamErr,
			})

			client.Close()
		}
	}

	// Analyze and report results
	t.Logf("Multi-Provider Streaming Comparison for prompt: %q", prompt)
	t.Logf("─────────────────────────────────────────────────────────")

	for _, result := range results {
		if result.Error != nil {
			t.Logf("%s: ERROR - %v", result.Name, result.Error)
		} else {
			t.Logf("%s:", result.Name)
			t.Logf("  Chunks: %d", result.ChunkCount)
			t.Logf("  Content length: %d chars", len(result.Content))
			if result.Usage != nil {
				t.Logf("  Usage: %d prompt + %d completion = %d total tokens",
					result.Usage.PromptTokens, result.Usage.CompletionTokens, result.Usage.TotalTokens)
			}
			t.Logf("  Response: %q", result.Content)
		}
		t.Logf("")
	}

	// Ensure at least one provider worked
	if len(results) == 0 {
		t.Fatal("No providers were tested")
	}

	successCount := 0
	for _, result := range results {
		if result.Error == nil && result.Content != "" {
			successCount++
		}
	}

	if successCount == 0 {
		t.Error("No providers successfully completed streaming")
	} else {
		t.Logf("Successfully tested streaming with %d/%d providers", successCount, len(results))
	}
}

// TestGoogleJSONModeIntegration validates that the JSON response format works correctly
// Run with: go test -tags=integration -v -run TestGoogleJSONMode
func TestGoogleJSONModeIntegration(t *testing.T) {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		t.Skip("GOOGLE_API_KEY not set, skipping integration test")
	}

	client, err := NewAIClient().
		WithGoogle(apiKey, "").
		WithDefaultProvider("google").
		WithDefaultModel("gemini-2.5-flash").
		Build()
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	ctx := context.Background()

	t.Run("json_object mode returns valid JSON", func(t *testing.T) {
		req := &types.CompletionRequest{
			Messages: []*types.Message{
				types.NewTextMessage(types.RoleSystem, "You are a JSON API. Always respond with valid JSON only."),
				types.NewTextMessage(types.RoleUser, `Count the words in this sentence: "Hello world this is a test"`),
			},
			MaxTokens:   256,
			Temperature: 0.1,
			ResponseFormat: &types.ResponseFormat{
				Type: "json_object",
			},
		}

		resp, err := client.Complete(ctx, req)
		if err != nil {
			t.Fatalf("Completion failed: %v", err)
		}

		responseText := resp.Message.GetText()
		t.Logf("Raw response: %q", responseText)

		// Verify response is NOT wrapped in markdown code blocks
		if strings.HasPrefix(responseText, "```") {
			t.Errorf("Response should not be wrapped in markdown code blocks, got: %s", responseText)
		}
		if strings.Contains(responseText, "```json") {
			t.Errorf("Response contains markdown JSON block, got: %s", responseText)
		}

		// Verify response is valid JSON
		var result map[string]interface{}
		if err := json.Unmarshal([]byte(responseText), &result); err != nil {
			t.Errorf("Response is not valid JSON: %v\nResponse: %s", err, responseText)
		}

		// Verify expected field exists
		if _, ok := result["word_count"]; !ok {
			t.Errorf("Expected 'word_count' field in response, got: %v", result)
		}

		t.Logf("Parsed JSON: %+v", result)
	})

	t.Run("json_object mode with array response", func(t *testing.T) {
		req := &types.CompletionRequest{
			Messages: []*types.Message{
				types.NewTextMessage(types.RoleUser, `List exactly 3 programming languages with their paradigms. Return as a JSON array:
[{"name": "<language>", "paradigm": "<paradigm>"}]`),
			},
			MaxTokens:   300,
			Temperature: 0.1,
			ResponseFormat: &types.ResponseFormat{
				Type: "json_object",
			},
		}

		resp, err := client.Complete(ctx, req)
		if err != nil {
			t.Fatalf("Completion failed: %v", err)
		}

		responseText := resp.Message.GetText()
		t.Logf("Raw response: %q", responseText)

		// Verify no markdown wrapping
		if strings.Contains(responseText, "```") {
			t.Errorf("Response contains markdown code blocks: %s", responseText)
		}

		// Verify valid JSON (could be array or object containing array)
		var result interface{}
		if err := json.Unmarshal([]byte(responseText), &result); err != nil {
			t.Errorf("Response is not valid JSON: %v\nResponse: %s", err, responseText)
		}

		t.Logf("Parsed JSON: %+v", result)
	})

	t.Run("without json_object mode may return markdown", func(t *testing.T) {
		// This test documents the behavior WITHOUT json_object mode
		// The response may or may not be wrapped in markdown
		req := &types.CompletionRequest{
			Messages: []*types.Message{
				types.NewTextMessage(types.RoleUser, `Return this exact JSON: {"test": true}`),
			},
			MaxTokens:   100,
			Temperature: 0.1,
			// Note: No ResponseFormat set
		}

		resp, err := client.Complete(ctx, req)
		if err != nil {
			t.Fatalf("Completion failed: %v", err)
		}

		responseText := resp.Message.GetText()
		t.Logf("Response without json_object mode: %q", responseText)

		// Just log the result - we're not asserting markdown wrapping
		// because behavior may vary
		if strings.Contains(responseText, "```") {
			t.Logf("Note: Response was wrapped in markdown (expected without json_object mode)")
		} else {
			t.Logf("Note: Response was NOT wrapped in markdown")
		}
	})
}

// TestGoogleJSONModeComplexStructure tests JSON mode with nested structures
func TestGoogleJSONModeComplexStructure(t *testing.T) {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		t.Skip("GOOGLE_API_KEY not set, skipping integration test")
	}

	client, err := NewAIClient().
		WithGoogle(apiKey, "").
		WithDefaultProvider("google").
		WithDefaultModel("gemini-2.5-flash").
		Build()
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	ctx := context.Background()

	req := &types.CompletionRequest{
		Messages: []*types.Message{
			types.NewTextMessage(types.RoleSystem, "You are a JSON API. Always respond with valid JSON only, no explanations."),
			types.NewTextMessage(types.RoleUser, `Analyze these 2 article headlines and rate their trending potential.

Article 1: "Breaking: Major Tech Company Announces Revolutionary AI Breakthrough"
Article 2: "Local Weather Report: Partly Cloudy Tomorrow"

Respond with this JSON structure:
{"articles": [{"id": 1, "title": "headline", "trending_score": 8.5, "trending_reason": "reason"}], "analysis_summary": "summary"}`),
		},
		MaxTokens:   800,
		Temperature: 0.2,
		ResponseFormat: &types.ResponseFormat{
			Type: "json_object",
		},
	}

	resp, err := client.Complete(ctx, req)
	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}

	responseText := resp.Message.GetText()
	t.Logf("Raw response: %q", responseText)

	// Verify no markdown wrapping
	if strings.HasPrefix(strings.TrimSpace(responseText), "```") {
		t.Errorf("Response starts with markdown code block: %s", responseText)
	}

	// Parse and validate structure
	var result struct {
		Articles []struct {
			ID             int     `json:"id"`
			Title          string  `json:"title"`
			TrendingScore  float64 `json:"trending_score"`
			TrendingReason string  `json:"trending_reason"`
		} `json:"articles"`
		AnalysisSummary string `json:"analysis_summary"`
	}

	if err := json.Unmarshal([]byte(responseText), &result); err != nil {
		t.Fatalf("Failed to parse response as expected structure: %v\nResponse: %s", err, responseText)
	}

	// Validate articles were analyzed
	if len(result.Articles) == 0 {
		t.Error("Expected at least one article in response")
	}

	for i, article := range result.Articles {
		if article.Title == "" {
			t.Errorf("Article %d has empty title", i)
		}
		if article.TrendingScore < 0 || article.TrendingScore > 10 {
			t.Errorf("Article %d has invalid trending score: %f", i, article.TrendingScore)
		}
	}

	t.Logf("Successfully parsed %d articles", len(result.Articles))
	t.Logf("Analysis summary: %s", result.AnalysisSummary)
}
