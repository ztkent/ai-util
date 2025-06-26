// Example usage of the OpenAI provider
//
// To run this example:
// 1. Set your OPENAI_API_KEY environment variable
// 2. go run examples/openai/openai_provider_example.go

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/ztkent/ai-util/providers/openai"
	"github.com/ztkent/ai-util/types"
)

func main() {
	// Get API key from environment
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("Please set OPENAI_API_KEY environment variable")
	}

	// Create and initialize the OpenAI provider
	provider := openai.NewProvider()
	config := &openai.Config{
		BaseConfig: types.BaseConfig{
			Provider: "openai",
			APIKey:   apiKey,
		},
	}

	if err := provider.Initialize(config); err != nil {
		log.Fatalf("Failed to initialize provider: %v", err)
	}

	fmt.Printf("Initialized OpenAI provider: %s\n", provider.GetName())

	ctx := context.Background()

	// List available models
	demonstrateListModels(ctx, provider)

	// Test basic chat completion
	demonstrateChat(ctx, provider)

	// Test streaming
	demonstrateStreaming(ctx, provider)

	// Test tools/function calling
	demonstrateTools(ctx, provider)

	// Clean up
	if err := provider.Close(); err != nil {
		log.Printf("Error closing provider: %v", err)
	}

	fmt.Println("\nExample completed successfully!")
}

func demonstrateListModels(ctx context.Context, provider *openai.Provider) {
	fmt.Printf("\n=== Listing Available Models ===\n")

	models, err := provider.GetModels(ctx)
	if err != nil {
		log.Printf("Failed to get models: %v", err)
		return
	}

	fmt.Printf("Available models (%d):\n", len(models))
	for i, model := range models {
		if i >= 10 { // Limit output for readability
			fmt.Printf("... and %d more models\n", len(models)-i)
			break
		}
		fmt.Printf("- %s: %s (max tokens: %d)\n", model.ID, model.Name, model.MaxTokens)
		fmt.Printf("  Capabilities: %v\n", model.Capabilities)
	}
}

func demonstrateChat(ctx context.Context, provider *openai.Provider) {
	fmt.Printf("\n=== Testing Chat Completion ===\n")

	messages := []*types.Message{
		{
			Role:     types.RoleSystem,
			TextData: "You are a helpful assistant that provides concise answers.",
		},
		{
			Role:     types.RoleUser,
			TextData: "Explain artificial intelligence in 2 sentences.",
		},
	}

	completionReq := &types.CompletionRequest{
		Messages:    messages,
		Model:       "gpt-4o-mini",
		MaxTokens:   100,
		Temperature: 0.7,
	}

	response, err := provider.Complete(ctx, completionReq)
	if err != nil {
		log.Printf("Failed to complete: %v", err)
		return
	}

	fmt.Printf("Response: %s\n", response.Message.TextData)
	fmt.Printf("Usage: Prompt=%d, Completion=%d, Total=%d tokens\n",
		response.Usage.PromptTokens,
		response.Usage.CompletionTokens,
		response.Usage.TotalTokens)
	fmt.Printf("Finish Reason: %s\n", response.FinishReason)
}

func demonstrateStreaming(ctx context.Context, provider *openai.Provider) {
	fmt.Printf("\n=== Testing Streaming ===\n")

	streamReq := &types.CompletionRequest{
		Messages: []*types.Message{
			{
				Role:     types.RoleUser,
				TextData: "Write a short poem about programming in Go.",
			},
		},
		Model:       "gpt-4o-mini",
		MaxTokens:   200,
		Temperature: 0.8,
	}

	fmt.Print("Streaming response: ")

	var totalUsage *types.Usage
	err := provider.Stream(ctx, streamReq, func(ctx context.Context, response *types.StreamResponse) error {
		if response.Delta != nil && response.Delta.TextData != "" {
			fmt.Print(response.Delta.TextData)
		}

		// Capture final usage stats
		if response.Usage != nil {
			totalUsage = response.Usage
		}

		return nil
	})

	if err != nil {
		log.Printf("Failed to stream: %v", err)
		return
	}

	fmt.Println()
	if totalUsage != nil {
		fmt.Printf("Streaming Usage: Prompt=%d, Completion=%d, Total=%d tokens\n",
			totalUsage.PromptTokens,
			totalUsage.CompletionTokens,
			totalUsage.TotalTokens)
	}
}

func demonstrateTools(ctx context.Context, provider *openai.Provider) {
	fmt.Printf("\n=== Testing Tools/Function Calling ===\n")

	// Define a weather tool
	weatherTool := types.Tool{
		Type: "function",
		Function: &types.ToolFunction{
			Name:        "get_weather",
			Description: "Get the current weather for a specific location",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"location": map[string]interface{}{
						"type":        "string",
						"description": "The city and state, e.g. San Francisco, CA",
					},
					"unit": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"celsius", "fahrenheit"},
						"description": "The unit of temperature",
					},
				},
				"required": []string{"location"},
			},
		},
	}

	// Define a calculator tool
	calculatorTool := types.Tool{
		Type: "function",
		Function: &types.ToolFunction{
			Name:        "calculate",
			Description: "Perform basic mathematical calculations",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"expression": map[string]interface{}{
						"type":        "string",
						"description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')",
					},
				},
				"required": []string{"expression"},
			},
		},
	}

	messages := []*types.Message{
		{
			Role:     types.RoleUser,
			TextData: "What's the weather like in San Francisco, CA? Also, what's 15 * 24?",
		},
	}

	toolReq := &types.CompletionRequest{
		Messages:    messages,
		Model:       "gpt-4o-mini",
		MaxTokens:   300,
		Temperature: 0.1,
		Tools:       []types.Tool{weatherTool, calculatorTool},
		ToolChoice:  "auto", // Let the model decide when to use tools
	}

	fmt.Println("Sending request with tools...")
	response, err := provider.Complete(ctx, toolReq)
	if err != nil {
		log.Printf("Failed to complete with tools: %v", err)
		return
	}

	fmt.Printf("Model response: %s\n", response.Message.TextData)

	// Handle tool calls
	if len(response.Message.ToolCalls) > 0 {
		fmt.Printf("\nModel wants to call %d tool(s):\n", len(response.Message.ToolCalls))

		// Add the assistant's message with tool calls first
		messages = append(messages, response.Message)

		// Process each tool call and add tool result messages
		for i, toolCall := range response.Message.ToolCalls {
			fmt.Printf("\nTool Call %d:\n", i+1)
			fmt.Printf("  ID: %s\n", toolCall.ID)
			fmt.Printf("  Function: %s\n", toolCall.Function.Name)
			fmt.Printf("  Arguments: %s\n", toolCall.Function.Arguments)

			// Simulate tool execution
			result := executeToolCall(toolCall)
			fmt.Printf("  Result: %s\n", result)

			// Add tool result to conversation
			messages = append(messages, &types.Message{
				Role: types.RoleTool,
				ToolResult: &types.ToolResult{
					ToolCallID: toolCall.ID,
					Content:    result,
				},
			})
		}

		// Get final response with tool results
		fmt.Println("\nGetting final response with tool results...")
		finalReq := &types.CompletionRequest{
			Messages:    messages,
			Model:       "gpt-4o-mini",
			MaxTokens:   200,
			Temperature: 0.1,
		}

		finalResponse, err := provider.Complete(ctx, finalReq)
		if err != nil {
			log.Printf("Failed to get final response: %v", err)
			return
		}

		fmt.Printf("Final response: %s\n", finalResponse.Message.TextData)
		fmt.Printf("Total Usage: Prompt=%d, Completion=%d, Total=%d tokens\n",
			finalResponse.Usage.PromptTokens,
			finalResponse.Usage.CompletionTokens,
			finalResponse.Usage.TotalTokens)
	}
}

// executeToolCall simulates executing a tool call and returns a result
func executeToolCall(toolCall types.ToolCall) string {
	switch toolCall.Function.Name {
	case "get_weather":
		// Parse arguments
		var args map[string]interface{}
		if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
			return fmt.Sprintf("Error parsing arguments: %v", err)
		}

		location, ok := args["location"].(string)
		if !ok {
			return "Error: location not provided"
		}

		unit := "fahrenheit"
		if u, ok := args["unit"].(string); ok {
			unit = u
		}

		// Simulate weather data
		temp := "72"
		if unit == "celsius" {
			temp = "22"
		}

		return fmt.Sprintf("The weather in %s is currently sunny with a temperature of %s°%s",
			location, temp, map[string]string{"celsius": "C", "fahrenheit": "F"}[unit])

	case "calculate":
		// Parse arguments
		var args map[string]interface{}
		if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
			return fmt.Sprintf("Error parsing arguments: %v", err)
		}

		expression, ok := args["expression"].(string)
		if !ok {
			return "Error: expression not provided"
		}

		// Simple calculation simulation (in real implementation, you'd use a proper math parser)
		switch expression {
		case "15 * 24":
			return "360"
		case "2 + 2":
			return "4"
		case "10 * 5":
			return "50"
		default:
			return fmt.Sprintf("Calculated result for '%s': [simulated result - implement proper math parser]", expression)
		}

	default:
		return fmt.Sprintf("Unknown tool: %s", toolCall.Function.Name)
	}
}
