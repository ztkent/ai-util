package aiutil

import (
	"fmt"

	"github.com/pkoukk/tiktoken-go"
	"github.com/sashabaranov/go-openai"

	// github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
	tiktoken_loader "github.com/pkoukk/tiktoken-go-loader"
)

// Estimate the number of tokens in the message using the OpenAI tokenizer
func EstimateMessageTokens(m openai.ChatCompletionMessage) (int, error) {
	content := m.Content
	for _, part := range m.MultiContent {
		if part.Type == openai.ChatMessagePartTypeText {
			content += part.Text
		} else if part.Type == openai.ChatMessagePartTypeImageURL {
			// This can vary depending on the image.
		}
	}

	// GPT3.5+ tokenizer
	encoding := "cl100k_base"
	// Avoid loading the dictionary at runtime, use offline loader
	tiktoken.SetBpeLoader(tiktoken_loader.NewOfflineLoader())
	tke, err := tiktoken.GetEncoding(encoding)
	if err != nil {
		return 0, fmt.Errorf("Failed to get encoding: %v", err)
	}

	// Tokenize and return the count
	tokens := tke.Encode(content, nil, nil)
	return len(tokens), nil
}
