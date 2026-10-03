// A tour of ai-util: chat, streaming, and a tool-calling agent.
//
//	OPENROUTER_API_KEY=... go run ./examples
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	aiutil "github.com/ztkent/ai-util"
)

const model = "poolside/laguna-s-2.1:free"

func main() {
	client := aiutil.New("") // reads OPENROUTER_API_KEY
	ctx := context.Background()

	chat(ctx, client)
	stream(ctx, client)
	agent(ctx, client)
}

func chat(ctx context.Context, client *aiutil.Client) {
	fmt.Println("== chat ==")
	resp, err := client.Chat(ctx, &aiutil.Request{
		Model:     model,
		Messages:  []aiutil.Message{aiutil.User("Explain goroutines in one sentence.")},
		MaxTokens: 100,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Message.Content)
	fmt.Printf("tokens: %d\n\n", resp.Usage.TotalTokens)
}

func stream(ctx context.Context, client *aiutil.Client) {
	fmt.Println("== stream ==")
	_, err := client.ChatStream(ctx, &aiutil.Request{
		Model:     model,
		Messages:  []aiutil.Message{aiutil.User("Write a haiku about Go.")},
		MaxTokens: 400,
	}, func(e aiutil.Event) error {
		if e.Type == aiutil.EventText {
			fmt.Print(e.Text)
		}
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println()
	fmt.Println()
}

func agent(ctx context.Context, client *aiutil.Client) {
	fmt.Println("== agent with tools ==")

	weather := aiutil.Tool{
		Name:        "get_weather",
		Description: "Get the current weather for a city.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string", "description": "city name"},
			},
			"required": []string{"city"},
		},
		Handler: func(ctx context.Context, args json.RawMessage) (string, error) {
			var in struct{ City string }
			if err := json.Unmarshal(args, &in); err != nil {
				return "", err
			}
			return fmt.Sprintf("%s: 21°C, sunny", in.City), nil
		},
	}

	a := aiutil.NewAgent(client, model,
		aiutil.WithSystem("You are a concise assistant. Use tools when needed."),
		aiutil.WithTools(weather),
	)

	resp, err := a.RunStream(ctx, "What's the weather in Paris?", func(e aiutil.Event) error {
		switch e.Type {
		case aiutil.EventText:
			fmt.Print(e.Text)
		case aiutil.EventToolCall:
			fmt.Printf("\n[tool] %s(%s)\n", e.ToolCall.Name, e.ToolCall.Arguments)
		case aiutil.EventToolResult:
			fmt.Printf("[result] %s\n", e.Result)
		}
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nfinal: %s\n", resp.Message.Content)
}
