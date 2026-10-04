# ai-util

Chat, stream, and run tool-calling agents.


No dependencies beyond the standard library.
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

`ChatStream` returns the same `*Response` as `Chat`, so you still get the full
message and usage once the stream ends.

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

`RunStream` is the same but emits Events. The conversation lives in
`agent.History()`; call `agent.Reset()` to clear it.

### Observing and accounting for calls

`NewAgent` accepts any `ChatClient`, so you can wrap a `*Client` to observe or
account for calls (for example, a cost ledger):

```go
type ChatClient interface {
    Chat(ctx context.Context, req *Request) (*Response, error)
    ChatStream(ctx context.Context, req *Request, onEvent func(Event) error) (*Response, error)
}
```

Per-turn hooks let you track usage and abort mid-loop:

```go
agent := aiutil.NewAgent(client, model,
    aiutil.WithOnUsage(func(u aiutil.Usage) error {
        ledger.Record(u)
        if ledger.OverBudget() {
            return errBudgetExceeded // aborts the run
        }
        return nil
    }),
    aiutil.WithOnTurn(func(resp *aiutil.Response) error { return nil }),
)
```

###
### Structured output

Ask for JSON conforming to a schema, and parse it with retries:

```go
schema := aiutil.Schema("answer", map[string]any{
    "type": "object",
    "properties": map[string]any{"n": map[string]any{"type": "integer"}},
})
var out struct{ N int }
_, err := agent.RunJSON(ctx, "give me n", schema, &out, aiutil.WithMaxAttempts(3))
```

The schema is applied per call, so it never leaks into the agent's state or
other runs. You can also set it on a single run:

```go
resp, err := agent.Run(ctx, "give me n", aiutil.WithSchema(schema))
```

`WithJSONMode()` asks for any JSON object; `Request.ResponseSchema` sets the
schema directly on a raw client call.

### Multimodal messages

```go
msg := aiutil.UserParts(
    aiutil.TextPart("what is in this image?"),
    aiutil.ImagePart("https://example.com/a.png"),
)
```

### Persisting conversations

`Message` and `ToolCall` are JSON-serializable, so history can be stored and
restored (for example, to replay a run):

```go
raw, _ := json.Marshal(agent.History())
// later
var msgs []aiutil.Message
json.Unmarshal(raw, &msgs)
agent := aiutil.NewAgent(client, model, aiutil.WithHistory(msgs...))
```

### Embeddings and model listing

```go
emb, err := client.Embed(ctx, &aiutil.EmbeddingRequest{Model: "...", Input: []string{"hello"}})
models, err := client.ListModels(ctx)
```

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
| `EventUsage` | Token usage for the call (`Usage`). |
| `EventDone` | Stream finished (`Response`). |

## Errors

Non-2xx responses and upstream failures return `*APIError`:

```go
if apiErr, ok := aiutil.IsAPIError(err); ok {
    fmt.Println(apiErr.StatusCode, apiErr.Message)
}
```

Transient failures (429, 5xx, network) are retried automatically with
exponential backoff, honoring `Retry-After`. Tune with `WithRetry`, or override
per request with `Request.Retry`.

`Agent.Run` returns `ErrMaxTurns` (detect with `errors.Is`) when it exceeds
`MaxTurns` without a final answer.

## Options

Client:

- `WithDefaultModel(model)`
- `WithBaseURL(url)`
- `WithHTTPClient(hc)`
- `WithRetry(policy)`

Agent:

- `WithSystem(prompt)`
- `WithTools(tools...)`
- `WithMaxTurns(n)`
- `WithMaxToolOutput(n)`
- `WithMaxContextTokens(n)`
- `WithOnUsage(fn)`
- `WithOnTurn(fn)`
- `WithHistory(msgs...)`


