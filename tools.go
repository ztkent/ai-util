package aiutil

import (
	"fmt"

	"github.com/pkoukk/tiktoken-go"
	"github.com/sashabaranov/go-openai"

	tiktoken_loader "github.com/pkoukk/tiktoken-go-loader"
)

// Estimate the number of tokens in the message using the OpenAI tokenizer
func EstimateMessageTokens(m openai.ChatCompletionMessage) (int, error) {
	content := m.Content
	for _, part := range m.MultiContent {
		if part.Type == openai.ChatMessagePartTypeText {
			content += part.Text
		} else if part.Type == openai.ChatMessagePartTypeImageURL {
			// TODO: Image token estimation is complex and varies.
		}
	}

	encoding := "cl100k_base"
	tiktoken.SetBpeLoader(tiktoken_loader.NewOfflineLoader())
	tke, err := tiktoken.GetEncoding(encoding)
	if err != nil {
		return 0, fmt.Errorf("Failed to get encoding: %v", err)
	}

	tokens := tke.Encode(content, nil, nil)
	return len(tokens), nil
}
