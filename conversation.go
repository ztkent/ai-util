package aiutil

import (
	"fmt"
	"sync"

	"github.com/google/uuid"
	"github.com/sashabaranov/go-openai"
)

type Conversation struct {
	Messages         []openai.ChatCompletionMessage
	TokenCount       int
	MaxTokens        int
	ResourcesEnabled bool
	id               uuid.UUID
	*sync.Mutex
}

// Start a new conversation with the system prompt
func NewConversation(systemPrompt string, maxTokens int, resourcesEnabled bool) *Conversation {
	if maxTokens <= 0 {
		maxTokens = DefaultMaxTokens // Use constant from connections.go
	}

	systemMessage := openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleSystem,
		Content: systemPrompt,
	}
	initialTokens, err := EstimateMessageTokens(systemMessage)
	if err != nil {
		initialTokens = 0
	}

	conv := &Conversation{
		Messages:         []openai.ChatCompletionMessage{systemMessage},
		TokenCount:       initialTokens,
		MaxTokens:        maxTokens,
		ResourcesEnabled: resourcesEnabled,
		id:               uuid.New(),
		Mutex:            &sync.Mutex{},
	}
	return conv
}

// Append adds a message, checking token limits.
func (c *Conversation) Append(m openai.ChatCompletionMessage) error {
	c.Lock()
	defer c.Unlock()

	tokCount, err := EstimateMessageTokens(m)
	if err != nil {
		return fmt.Errorf("failed to estimate message tokens: %w", err)
	}

	if c.TokenCount+tokCount > c.MaxTokens {
		// Context pruning strategy could be implemented here if desired
		return fmt.Errorf("max conversation tokens exceeded: adding %d tokens to current %d would exceed limit %d", tokCount, c.TokenCount, c.MaxTokens)
	}

	c.TokenCount += tokCount
	c.Messages = append(c.Messages, m)
	return nil
}

// RemoveLastMessageIfRole removes the last message if it matches the given role.
// Useful for cleaning up incomplete request/response pairs on error.
func (c *Conversation) RemoveLastMessageIfRole(role string) {
	c.Lock()
	defer c.Unlock()

	if len(c.Messages) > 0 {
		lastMsg := c.Messages[len(c.Messages)-1]
		if lastMsg.Role == role {
			tokCount, err := EstimateMessageTokens(lastMsg)
			if err == nil {
				c.TokenCount -= tokCount
				if c.TokenCount < 0 {
					c.TokenCount = 0
				}
			}
			c.Messages = c.Messages[:len(c.Messages)-1]
		}
	}
}

// SeedConversation adds example request/response pairs.
func (c *Conversation) SeedConversation(requestResponseMap map[string]string) error {
	for user, response := range requestResponseMap {
		err := c.Append(openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: user,
		})
		if err != nil {
			return fmt.Errorf("failed to seed user message (%s): %w", user, err)
		}
		err = c.Append(openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleAssistant,
			Content: response,
		})
		if err != nil {
			c.RemoveLastMessageIfRole(openai.ChatMessageRoleUser)
			return fmt.Errorf("failed to seed assistant response for user message (%s): %w", user, err)
		}
	}
	return nil
}

// AddReference adds a system message containing reference material.
func (c *Conversation) AddReference(id string, content string) error {
	refContent := fmt.Sprintf("<Reference id=\"%s\">\n%s\n</Reference>", id, content)

	message := openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleSystem,
		Content: refContent,
		// Name: "ReferenceMaterial", // Optional name
	}

	err := c.Append(message)
	if err != nil {
		return fmt.Errorf("failed to add reference '%s': %w", id, err)
	}
	return nil
}
