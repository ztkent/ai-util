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
	MaxTokens        int  // Max tokens for the *conversation history*, not response generation
	ResourcesEnabled bool // Keep for now, might relate to how references are handled
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
	// Estimate initial system prompt tokens
	initialTokens, err := EstimateMessageTokens(systemMessage)
	if err != nil {
		// Handle error? Log? For now, assume 0 if estimation fails.
		fmt.Printf("Warning: Failed to estimate initial system prompt tokens: %v\n", err)
		initialTokens = 0
	}
	if initialTokens > maxTokens {
		// System prompt itself exceeds limit, maybe return error or truncated conversation?
		// For now, create it anyway but it will fail on first Append.
		fmt.Printf("Warning: System prompt token count (%d) exceeds MaxTokens (%d)\n", initialTokens, maxTokens)
	}

	conv := &Conversation{
		Messages:         []openai.ChatCompletionMessage{systemMessage},
		TokenCount:       initialTokens, // Initialize token count
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

	// Check if adding this message exceeds the limit
	if c.TokenCount+tokCount > c.MaxTokens {
		// TODO: Implement context pruning strategy here if desired (e.g., remove oldest messages)
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
			// Re-estimate tokens of the removed message to subtract
			tokCount, err := EstimateMessageTokens(lastMsg)
			if err == nil { // Only adjust count if estimation succeeds
				c.TokenCount -= tokCount
				if c.TokenCount < 0 { // Avoid negative count
					c.TokenCount = 0
				}
			} else {
				fmt.Printf("Warning: Failed to estimate tokens for removed message: %v\n", err)
				// Token count might become inaccurate here. Consider recalculating all messages.
			}
			c.Messages = c.Messages[:len(c.Messages)-1]
		}
	}
}

// SeedConversation adds example request/response pairs.
func (c *Conversation) SeedConversation(requestResponseMap map[string]string) error {
	// No lock needed here as Append handles locking internally
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
			// Attempt to clean up the user message added just before
			c.RemoveLastMessageIfRole(openai.ChatMessageRoleUser)
			return fmt.Errorf("failed to seed assistant response for user message (%s): %w", user, err)
		}
	}
	return nil
}

// AddReference adds a system message containing reference material.
func (c *Conversation) AddReference(id string, content string) error {
	// Build the reference message content
	// Using MultiContent might not be universally supported or interpreted correctly by all models.
	// A simpler approach might be a single text block.
	// Let's switch to a simpler text format for broader compatibility.
	refContent := fmt.Sprintf("<Reference id=\"%s\">\n%s\n</Reference>", id, content)

	// Consider adding a specific Name or Role if needed, but System role is often appropriate.
	message := openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleSystem, // Or maybe a custom role/name if models support it
		Content: refContent,
		// Name: "ReferenceMaterial", // Optional name
	}

	// Append handles locking and token counting
	err := c.Append(message)
	if err != nil {
		return fmt.Errorf("failed to add reference '%s': %w", id, err)
	}
	return nil
}
