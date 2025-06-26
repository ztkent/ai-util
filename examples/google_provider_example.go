// Example usage of the Google AI provider
//
// To run this example:
// 1. Set your GOOGLE_API_KEY environment variable
// 2. go run examples/google_provider_example.go

package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/ztkent/ai-util/providers/google"
	"github.com/ztkent/ai-util/types"
)

func main() {
	// Get API key from environment
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		log.Fatal("Please set GOOGLE_API_KEY environment variable")
	}

	// Create and initialize the Google provider
	provider := google.NewProvider()
	config := &google.Config{
		BaseConfig: types.BaseConfig{
			Provider: "google",
			APIKey:   apiKey,
		},
	}

	if err := provider.Initialize(config); err != nil {
		log.Fatalf("Failed to initialize provider: %v", err)
	}

	fmt.Printf("Initialized Google AI provider: %s\n", provider.GetName())

	// List available models
	ctx := context.Background()
	models, err := provider.GetModels(ctx)
	if err != nil {
		log.Fatalf("Failed to get models: %v", err)
	}

	fmt.Printf("\nAvailable models (%d):\n", len(models))
	for _, model := range models {
		fmt.Printf("- %s: %s (max tokens: %d)\n", model.ID, model.Name, model.MaxTokens)
	}

	// Test completion
	fmt.Printf("\n--- Testing Completion ---\n")
	messages := []*types.Message{
		{
			Role:     types.RoleUser,
			TextData: "Explain AI in one sentence.",
		},
	}

	completionReq := &types.CompletionRequest{
		Messages:    messages,
		Model:       "gemini-2.5-flash",
		MaxTokens:   100,
		Temperature: 0.7,
	}

	response, err := provider.Complete(ctx, completionReq)
	if err != nil {
		log.Fatalf("Failed to complete: %v", err)
	}

	fmt.Printf("Response: %s\n", response.Message.TextData)
	fmt.Printf("Usage: %+v\n", response.Usage)

	// Test streaming
	fmt.Printf("\n--- Testing Streaming ---\n")
	streamReq := &types.CompletionRequest{
		Messages: []*types.Message{
			{
				Role:     types.RoleUser,
				TextData: "Write a short poem about programming.",
			},
		},
		Model:       "gemini-2.5-flash",
		MaxTokens:   200,
		Temperature: 0.8,
	}

	fmt.Print("Streaming response: ")
	err = provider.Stream(ctx, streamReq, func(ctx context.Context, response *types.StreamResponse) error {
		fmt.Print(response.Delta.TextData)
		return nil
	})
	if err != nil {
		log.Fatalf("Failed to stream: %v", err)
	}
	fmt.Println()

	// Test token estimation
	fmt.Printf("\n--- Testing Token Estimation ---\n")
	tokens, err := provider.EstimateTokens(ctx, messages, "gemini-2.5-flash")
	if err != nil {
		log.Fatalf("Failed to estimate tokens: %v", err)
	}
	fmt.Printf("Estimated tokens: %d\n", tokens)

	// Clean up
	if err := provider.Close(); err != nil {
		log.Printf("Error closing provider: %v", err)
	}

	fmt.Println("\nExample completed successfully!")
}
