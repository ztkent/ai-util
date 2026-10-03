# ai-util

Chat, stream, and run tool-calling agents.
No dependencies beyond the standard library.

## Install

```bash
go get github.com/ztkent/ai-util
```

Set `OPENROUTER_API_KEY`. Any OpenRouter model works.

## Chat

```go
client := aiutil.New("")

resp, err := client.Chat(ctx, &aiutil.Request{
    Model:     "poolside/laguna-s-2.1:free",
    Messages:  []aiutil.Message{aiutil.User("Explain goroutines in one sentence.")},
    MaxTokens: 200,
})
fmt.Println(resp.Message.Content, resp.Usage.TotalTokens)
```

## Stream

```go
resp, err := client.ChatStream(ctx, req, func(e aiutil.Event) error {
    if e.Type == aiutil.EventText {
        fmt.Print(e.Text)
    }
    return nil
})
```

`ChatStream` returns the same `*Response` as `Chat`, so you get the full message
and usage after the stream ends.

## Agent

An `Agent` runs the tool loop for you: it calls the model, executes any tools,
feeds the results back, and repeats until the model answers.

```go
weather := aiutil.Tool{
    Name:        "get_weather",
    Description: "Get the current weather for a city.",
    Parameters: map[string]any{
        "type": "object",
        "properties": map[string]any{
            "city": map[string]any{"type": "string"},
        },
        "required": []string{"city"},
    },
    Handler: func(ctx context.Context, args json.RawMessage) (string, error) {
        var in struct{ City string }
        json.Unmarshal(args, &in)
        return in.City + ": 21°C, sunny", nil
    },
}

agent := aiutil.NewAgent(client, "poolside/laguna-s-2.1:free",
    aiutil.WithSystem("Be concise."),
    aiutil.WithTools(weather),
)

resp, err := agent.Run(ctx, "What's the weather in Paris?")
```

`RunStream` is the same but emits `Event`s. The conversation is kept in
`agent.History()`; call `agent.Reset()` to clear it.

## Request fields

| Field | Notes |
|---|---|
| `Model` | Falls back to `WithDefaultModel`. |
| `Messages` | Required. |
| `Tools` | Tools the model may call. |
| `ToolChoice` | `"auto"`, `"none"`, `"required"`, or a specific function. |
| `Temperature`, `TopP` | `*float64` so `0` is a real value, not "unset". |
| `MaxTokens` | Output cap. |
| `Stop` | Stop sequences. |
| `JSONMode` | Ask for a JSON object. |

## Events

| Type | Meaning |
|---|---|
| `EventText` | A chunk of assistant text (`Text`). |
| `EventToolCall` | A completed tool call (`ToolCall`). |
| `EventToolResult` | A tool's output (`ToolCall`, `Result`). |
| `EventDone` | Stream finished (`Response`). |

## Errors

Non-2xx responses and upstream failures return `*APIError`:

```go
if apiErr, ok := aiutil.IsAPIError(err); ok {
    fmt.Println(apiErr.StatusCode, apiErr.Message)
}
```

Transient failures (429, 5xx, network) are retried automatically with
exponential backoff, honoring `Retry-After`. Tune with `WithRetry`.

## Options

- `WithDefaultModel(model)`
- `WithBaseURL(url)`
- `WithHTTPClient(hc)`
- `WithRetry(policy)`

## Testing

```bash
go test ./...                                          # unit tests (mock server)
OPENROUTER_API_KEY=... go test -tags=integration ./... # live, prefer free models
```

Override the integration model with `OPENROUTER_TEST_MODEL`.
