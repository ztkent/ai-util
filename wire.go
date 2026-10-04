package aiutil

// This file contains the OpenAI-compatible wire format used by OpenRouter.
// Keeping it isolated means the rest of the package never touches raw JSON
// field names.

type wireRequest struct {
	Model          string          `json:"model"`
	Messages       []wireMessage   `json:"messages"`
	Tools          []wireTool      `json:"tools,omitempty"`
	ToolChoice     any             `json:"tool_choice,omitempty"`
	Temperature    *float64        `json:"temperature,omitempty"`
	MaxTokens      int             `json:"max_tokens,omitempty"`
	TopP           *float64        `json:"top_p,omitempty"`
	Stop           []string        `json:"stop,omitempty"`
	Stream         bool            `json:"stream,omitempty"`
	StreamOptions  *streamOptions  `json:"stream_options,omitempty"`
	ResponseFormat *responseFormat `json:"response_format,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type responseFormat struct {
	Type       string         `json:"type"`
	JSONSchema map[string]any `json:"json_schema,omitempty"`
}

type wireMessage struct {
	Role       string         `json:"role"`
	Content    any            `json:"content"`
	ToolCalls  []wireToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	Name       string         `json:"name,omitempty"`
}

type wireContentPart struct {
	Type     string        `json:"type"`
	Text     string        `json:"text,omitempty"`
	ImageURL *wireImageURL `json:"image_url,omitempty"`
}

type wireImageURL struct {
	URL string `json:"url"`
}

type wireToolCall struct {
	Index    *int         `json:"index,omitempty"`
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Function wireFunction `json:"function"`
}

type wireFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type wireTool struct {
	Type     string      `json:"type"`
	Function wireToolDef `json:"function"`
}

type wireToolDef struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

type wireUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type wireError struct {
	Code    any    `json:"code"`
	Message string `json:"message"`
}

type wireResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Message      wireMessage `json:"message"`
		FinishReason string      `json:"finish_reason"`
	} `json:"choices"`
	Usage *wireUsage `json:"usage"`
	Error *wireError `json:"error"`
}

type wireChunk struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Delta        wireMessage `json:"delta"`
		FinishReason string      `json:"finish_reason"`
	} `json:"choices"`
	Usage *wireUsage `json:"usage"`
	Error *wireError `json:"error"`
}

func toWireMessages(msgs []Message) []wireMessage {
	out := make([]wireMessage, len(msgs))
	for i, m := range msgs {
		out[i] = wireMessage{
			Role:       string(m.Role),
			Content:    wireContent(m),
			ToolCallID: m.ToolCallID,
			Name:       m.Name,
		}
		for _, tc := range m.ToolCalls {
			out[i].ToolCalls = append(out[i].ToolCalls, wireToolCall{
				ID:       tc.ID,
				Type:     "function",
				Function: wireFunction{Name: tc.Name, Arguments: tc.Arguments},
			})
		}
	}
	return out
}

// wireContent renders a message's content as either a plain string or a list
// of multimodal parts.
func wireContent(m Message) any {
	if len(m.Parts) == 0 {
		return m.Content
	}
	parts := make([]wireContentPart, len(m.Parts))
	for i, p := range m.Parts {
		switch p.Type {
		case "image_url":
			parts[i] = wireContentPart{Type: "image_url", ImageURL: &wireImageURL{URL: p.ImageURL}}
		default:
			parts[i] = wireContentPart{Type: "text", Text: p.Text}
		}
	}
	return parts
}

func toWireTools(tools []Tool) []wireTool {
	out := make([]wireTool, len(tools))
	for i, t := range tools {
		out[i] = wireTool{
			Type: "function",
			Function: wireToolDef{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.schema(),
			},
		}
	}
	return out
}

func fromWireMessage(m wireMessage) Message {
	msg := Message{
		Role:       Role(m.Role),
		ToolCallID: m.ToolCallID,
		Name:       m.Name,
	}
	switch c := m.Content.(type) {
	case string:
		msg.Content = c
	case []any:
		for _, raw := range c {
			part, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			p := ContentPart{}
			if t, ok := part["type"].(string); ok {
				p.Type = t
			}
			if t, ok := part["text"].(string); ok {
				p.Text = t
			}
			if iu, ok := part["image_url"].(map[string]any); ok {
				if u, ok := iu["url"].(string); ok {
					p.ImageURL = u
				}
			}
			msg.Parts = append(msg.Parts, p)
		}
	}
	for _, tc := range m.ToolCalls {
		msg.ToolCalls = append(msg.ToolCalls, ToolCall{
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: tc.Function.Arguments,
		})
	}
	return msg
}
