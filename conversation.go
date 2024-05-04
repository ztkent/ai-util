package aiutil

import (
	"fmt"
	"strings"
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
		return fmt.Errorf("Max tokens exceeded [ %d > %d ]", attemptedTokens, c.MaxTokens)
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

// Add a reference to the conversation
func (c *Conversation) AddReference(id string, content string) error {
	// Build the response message
	messageParts := make([]openai.ChatMessagePart, 0)
	messageParts = append(messageParts, openai.ChatMessagePart{
		Type: openai.ChatMessagePartTypeText,
		Text: "<Id>" + id + "</Id>",
	})
	messageParts = append(messageParts, openai.ChatMessagePart{
		Type: openai.ChatMessagePartTypeText,
		Text: "<Content> " + content + " </Content>",
	})
	return c.Append(openai.ChatCompletionMessage{
		Name:         "Reference",
		Role:         openai.ChatMessageRoleSystem,
		MultiContent: messageParts,
	})
}

func (c *Conversation) History() string {
	// First message is a system prompt, last is the current user prompt
	if len(c.Messages) < 3 {
		return ""
	}

	var str strings.Builder
	str.WriteString(fmt.Sprintf("History: \n"))
	for _, m := range c.Messages[1 : len(c.Messages)-2] {
		str.WriteString(fmt.Sprintf("Role: %s, Content: %s\n", m.Role, m.Content))
	}
	return str.String()
}
