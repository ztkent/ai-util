//go:build integration
// +build integration

package google

import (
	"context"
	"os"
	"testing"

	"github.com/ztkent/ai-util/types"
)

// Run with: go test -tags=integration -v ./providers/google/...

func TestGemmaModelsIntegration(t *testing.T) {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		t.Skip("GOOGLE_API_KEY not set, skipping integration test")
	}

	provider := NewProvider()
	config := &Config{
		BaseConfig: types.BaseConfig{
			Provider: "google",
			APIKey:   apiKey,
		},
	}

	if err := provider.Initialize(config); err != nil {
		t.Fatalf("Failed to initialize provider: %v", err)
	}
	defer provider.Close()

	gemmaModels := []struct {
		id          string
		name        string
		description string
	}{
		{"gemma-3-27b-it", "Gemma 3 27B", "Best for complex reasoning and chat"},
		{"gemma-3-12b-it", "Gemma 3 12B", "High performance for laptops/desktops"},
		{"gemma-3-4b-it", "Gemma 3 4B", "Balanced for efficiency and mobile"},
		{"gemma-3-1b-it", "Gemma 3 1B", "Ultra-efficient for text-only tasks"},
	}

	ctx := context.Background()

	for _, model := range gemmaModels {
		t.Run(model.id, func(t *testing.T) {
			req := &types.CompletionRequest{
				Model: model.id,
				Messages: []*types.Message{
					types.NewTextMessage(types.RoleUser, "Say hello in exactly 5 words"),
				},
				MaxTokens:   50,
				Temperature: 0.7,
			}

			resp, err := provider.Complete(ctx, req)
			if err != nil {
				t.Fatalf("Completion failed for %s (%s): %v", model.name, model.description, err)
			}

			if resp.Message == nil || resp.Message.GetText() == "" {
				t.Errorf("Expected non-empty response from %s", model.name)
			}

			if resp.Provider != "google" {
				t.Errorf("Expected provider 'google', got '%s'", resp.Provider)
			}

			if resp.Model != model.id {
				t.Errorf("Expected model '%s', got '%s'", model.id, resp.Model)
			}

			t.Logf("%s (%s) Response: %s", model.name, model.description, resp.Message.GetText())
			if resp.Usage != nil {
				t.Logf("%s Usage: %d prompt + %d completion = %d total tokens",
					model.name,
					resp.Usage.PromptTokens,
					resp.Usage.CompletionTokens,
					resp.Usage.TotalTokens)
			}
		})
	}
}
