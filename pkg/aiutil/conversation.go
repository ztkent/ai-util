package aiutil

import (
	"fmt"
	"sync"

	"github.com/google/uuid"
	"github.com/sashabaranov/go-openai"
)

type Conversation struct {
	Messages   []openai.ChatCompletionMessage
	TokenCount int
	MaxTokens  int
	RagEnabled bool
	id         uuid.UUID
	*sync.Mutex
}

const (
	/*
		04/12/2024 Pricing:
		gpt-3.5-turbo-0125			Input: $0.50 / 1M tokens 	Output: $1.50 / 1M tokens
		gpt-4-turbo-2024-04-09		Input: $10.00 / 1M tokens	Output: $30.00 / 1M tokens
		Mixtral-8x7B-Instruct-v0.1	Input: 0.50 / 1M tokens 	Output: 0.50 / 1M tokens
	*/
	DefaultMaxTokens = 100000 // $0.05, $0.15 | $1.00, $3.00
)

// Start a new conversation with the system prompt
// A system prompt defines the initial context of the conversation
// This includes the persona of the bot and any information that you want to provide to the model.
func NewConversation(systemPrompt string, maxTokens int, ragEnabled bool) *Conversation {
	if maxTokens == 0 {
		maxTokens = DefaultMaxTokens
	}
	conv := &Conversation{
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: systemPrompt,
			},
		},
		MaxTokens:  maxTokens,
		RagEnabled: ragEnabled,
		id:         uuid.New(),
		Mutex:      &sync.Mutex{},
	}
	return conv
}

func (c *Conversation) Append(m openai.ChatCompletionMessage) error {
	c.Lock()
	defer c.Unlock()
	tokCount, err := EstimateMessageTokens(m)
	if err != nil {
		return err
	}
	attemptedTokens := c.TokenCount + tokCount
	if attemptedTokens > c.MaxTokens {
		return fmt.Errorf("Max tokens exceeded")
	}
	c.TokenCount += tokCount
	c.Messages = append(c.Messages, m)
	return nil
}

func (c *Conversation) SeedConversation(requestResponseMap map[string]string) {
	// Seed the conversation with some example prompts and responses
	for user, response := range requestResponseMap {
		c.Append(openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: user,
		})
		c.Append(openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleAssistant,
			Content: response,
		})
	}
}
