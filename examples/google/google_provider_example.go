// Example usage of the Google AI provider
//
// To run this example:
// 1. Set your GOOGLE_API_KEY environment variable
// 2. go run examples/google/google_provider_example.go

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

	// Test tool calling
	fmt.Printf("\n--- Testing Tool Calling ---\n")
	demonstrateTools(ctx, provider)

	// Clean up
	if err := provider.Close(); err != nil {
		log.Printf("Error closing provider: %v", err)
	}

	fmt.Println("\nExample completed successfully!")
}

func demonstrateTools(ctx context.Context, provider *google.Provider) {
	fmt.Printf("Testing Tools/Function Calling with Google AI\n")

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
		Model:       "gemini-2.5-flash", // Use a model that supports tools
		MaxTokens:   300,
		Temperature: 0.1,
		Tools:       []types.Tool{weatherTool, calculatorTool},
	}

	fmt.Println("Sending request with tools...")
	response, err := provider.Complete(ctx, toolReq)
	if err != nil {
		log.Printf("Failed to complete with tools: %v", err)
		return
	}

	if response.Message.TextData != "" {
		fmt.Printf("Model response: %s\n", response.Message.TextData)
	} else {
		fmt.Printf("Model response: (no text, making tool calls only)\n")
	}

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
			fmt.Printf("  Arguments: %v\n", toolCall.Args)

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
			Model:       "gemini-2.5-flash",
			MaxTokens:   200,
			Temperature: 0.1,
		}

		finalResponse, err := provider.Complete(ctx, finalReq)
		if err != nil {
			log.Printf("Failed to get final response: %v", err)
			return
		}

		fmt.Printf("Final response: %s\n", finalResponse.Message.TextData)
	} else {
		fmt.Println("No tool calls were made by the model.")
	}
}

// executeToolCall simulates executing a tool call and returns a result
func executeToolCall(toolCall types.ToolCall) string {
	switch toolCall.Function.Name {
	case "get_weather":
		location, ok := toolCall.Args["location"].(string)
		if !ok {
			return "Error: location not provided"
		}

		unit := "fahrenheit"
		if u, ok := toolCall.Args["unit"].(string); ok {
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
		expression, ok := toolCall.Args["expression"].(string)
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

	case "run_code":
		// Handle when Google AI calls run_code instead of calculate
		code, ok := toolCall.Args["code"].(string)
		if !ok {
			return "Error: code not provided"
		}

		// Extract and calculate mathematical expressions from code
		switch code {
		case "print(15 * 24)":
			return "360"
		case "print(2 + 2)":
			return "4"
		case "print(10 * 5)":
			return "50"
		}

		return fmt.Sprintf("Executed code '%s': [simulated result - implement proper code execution]", code)

	default:
		return fmt.Sprintf("Unknown tool: %s", toolCall.Function.Name)
	}
}
