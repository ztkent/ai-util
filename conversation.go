package aiutil

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/ztkent/ai-util/types"
)

// Conversation represents a conversation with message history and management
type Conversation struct {
	ID              string                 `json:"id"`
	Messages        []*types.Message       `json:"messages"`
	MaxTokens       int                    `json:"max_tokens"`
	CurrentTokens   int                    `json:"current_tokens"`
	CreatedAt       time.Time              `json:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at"`
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
	client          *Client
	estimatedTokens int
	mu              sync.RWMutex
}

// ConversationConfig holds configuration for creating a conversation
type ConversationConfig struct {
	SystemPrompt   string                 `json:"system_prompt,omitempty"`
	MaxTokens      int                    `json:"max_tokens,omitempty"`
	Model          string                 `json:"model,omitempty"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
	AutoTruncate   bool                   `json:"auto_truncate,omitempty"`
	PreserveSystem bool                   `json:"preserve_system,omitempty"` // Keep system message when truncating
}

// NewConversation creates a new conversation with optional system prompt
func (c *Client) NewConversation(config *ConversationConfig) *Conversation {
	if config == nil {
		config = &ConversationConfig{
			MaxTokens:      4096,
			AutoTruncate:   true,
			PreserveSystem: true,
		}
	}

	if config.MaxTokens <= 0 {
		config.MaxTokens = 4096
	}

	conv := &Conversation{
		ID:        uuid.New().String(),
		Messages:  make([]*types.Message, 0),
		MaxTokens: config.MaxTokens,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		Metadata:  config.Metadata,
		client:    c,
	}

	// Add system message if provided
	if config.SystemPrompt != "" {
		systemMsg := types.NewTextMessage(types.RoleSystem, config.SystemPrompt)
		conv.AddMessage(systemMsg)
	}

	return conv
}

// AddMessage adds a message to the conversation
func (c *Conversation) AddMessage(message *types.Message) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if message.Timestamp.IsZero() {
		message.Timestamp = time.Now()
	}

	c.Messages = append(c.Messages, message)
	c.UpdatedAt = time.Now()

	// Update token count estimation
	if c.client != nil {
		// Use a default model for estimation if none specified
		model := c.client.defaultConfig.DefaultModel
		if model == "" {
			model = "gpt-3.5-turbo" // Fallback
		}

		tokens, err := c.client.EstimateTokens(context.Background(), []*types.Message{message}, model)
		if err == nil {
			c.estimatedTokens += tokens
		}
	}

	return nil
}

// AddUserMessage adds a user message to the conversation
func (c *Conversation) AddUserMessage(text string) error {
	message := types.NewTextMessage(types.RoleUser, text)
	return c.AddMessage(message)
}

// AddAssistantMessage adds an assistant message to the conversation
func (c *Conversation) AddAssistantMessage(text string) error {
	message := types.NewTextMessage(types.RoleAssistant, text)
	return c.AddMessage(message)
}

// AddSystemMessage adds a system message to the conversation
func (c *Conversation) AddSystemMessage(text string) error {
	message := types.NewTextMessage(types.RoleSystem, text)
	return c.AddMessage(message)
}

// GetMessages returns a copy of all messages
func (c *Conversation) GetMessages() []*types.Message {
	c.mu.RLock()
	defer c.mu.RUnlock()

	messages := make([]*types.Message, len(c.Messages))
	copy(messages, c.Messages)
	return messages
}

// GetLastMessage returns the last message in the conversation
func (c *Conversation) GetLastMessage() *types.Message {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if len(c.Messages) == 0 {
		return nil
	}
	return c.Messages[len(c.Messages)-1]
}

// GetMessagesByRole returns messages filtered by role
func (c *Conversation) GetMessagesByRole(role types.Role) []*types.Message {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var filtered []*types.Message
	for _, msg := range c.Messages {
		if msg.Role == role {
			filtered = append(filtered, msg)
		}
	}
	return filtered
}

// TruncateToFit ensures the conversation fits within token limits
func (c *Conversation) TruncateToFit(ctx context.Context, model string, preserveSystem bool) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.client == nil {
		return types.NewError(types.ErrCodeInvalidConfig, "no client available for token estimation", "")
	}

	for {
		tokens, err := c.client.EstimateTokens(ctx, c.Messages, model)
		if err != nil {
			return err
		}

		if tokens <= c.MaxTokens {
			c.estimatedTokens = tokens
			break
		}

		// Remove messages from the middle, preserving system message if requested
		if err := c.removeOldestNonSystemMessage(preserveSystem); err != nil {
			return err
		}

		if len(c.Messages) == 0 || (preserveSystem && len(c.Messages) == 1) {
			return types.NewError(types.ErrCodeTokenLimitExceeded,
				"cannot fit conversation within token limit", "")
		}
	}

	return nil
}

// removeOldestNonSystemMessage removes the oldest non-system message
func (c *Conversation) removeOldestNonSystemMessage(preserveSystem bool) error {
	for i, msg := range c.Messages {
		if !preserveSystem || msg.Role != types.RoleSystem {
			c.Messages = append(c.Messages[:i], c.Messages[i+1:]...)
			return nil
		}
	}
	return fmt.Errorf("no removable messages found")
}

// Clear removes all messages from the conversation
func (c *Conversation) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.Messages = make([]*types.Message, 0)
	c.estimatedTokens = 0
	c.UpdatedAt = time.Now()
}

// Send sends a user message and gets a response
func (c *Conversation) Send(ctx context.Context, userMessage string, model string) (*types.CompletionResponse, error) {
	// Add user message
	if err := c.AddUserMessage(userMessage); err != nil {
		return nil, err
	}

	// Prepare request
	req := &types.CompletionRequest{
		Messages: c.GetMessages(),
		Model:    model,
	}

	// Send completion request
	resp, err := c.client.Complete(ctx, req)
	if err != nil {
		return nil, err
	}

	// Add assistant response to conversation
	if resp.Message != nil {
		if err := c.AddMessage(resp.Message); err != nil {
			return nil, err
		}
	}

	return resp, nil
}

// SendStream sends a user message and streams the response
func (c *Conversation) SendStream(ctx context.Context, userMessage string, model string, callback types.StreamCallback) error {
	// Add user message
	if err := c.AddUserMessage(userMessage); err != nil {
		return err
	}

	// Prepare request
	req := &types.CompletionRequest{
		Messages: c.GetMessages(),
		Model:    model,
		Stream:   true,
	}

	// Collect streaming response for conversation history
	var fullResponse string
	wrappedCallback := func(ctx context.Context, response *types.StreamResponse) error {
		if response.Delta != nil && response.Delta.TextData != "" {
			fullResponse += response.Delta.TextData
		}

		// Call the original callback
		if err := callback(ctx, response); err != nil {
			return err
		}

		// Add complete response to conversation when finished
		if response.FinishReason != "" && fullResponse != "" {
			assistantMsg := types.NewTextMessage(types.RoleAssistant, fullResponse)
			c.AddMessage(assistantMsg)
		}

		return nil
	}

	return c.client.Stream(ctx, req, wrappedCallback)
}

// EstimateTokens estimates the current token count of the conversation
func (c *Conversation) EstimateTokens(ctx context.Context, model string) (int, error) {
	if c.client == nil {
		return 0, types.NewError(types.ErrCodeInvalidConfig, "no client available for token estimation", "")
	}

	return c.client.EstimateTokens(ctx, c.GetMessages(), model)
}

// GetTokenCount returns the last estimated token count
func (c *Conversation) GetTokenCount() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.estimatedTokens
}

// Clone creates a copy of the conversation
func (c *Conversation) Clone() *Conversation {
	c.mu.RLock()
	defer c.mu.RUnlock()

	messages := make([]*types.Message, len(c.Messages))
	copy(messages, c.Messages)

	metadata := make(map[string]interface{})
	for k, v := range c.Metadata {
		metadata[k] = v
	}

	return &Conversation{
		ID:              uuid.New().String(),
		Messages:        messages,
		MaxTokens:       c.MaxTokens,
		CurrentTokens:   c.CurrentTokens,
		CreatedAt:       time.Now(),
		UpdatedAt:       time.Now(),
		Metadata:        metadata,
		client:          c.client,
		estimatedTokens: c.estimatedTokens,
	}
}

// Export exports the conversation to a JSON-serializable format
func (c *Conversation) Export() map[string]interface{} {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return map[string]interface{}{
		"id":               c.ID,
		"messages":         c.Messages,
		"max_tokens":       c.MaxTokens,
		"current_tokens":   c.CurrentTokens,
		"estimated_tokens": c.estimatedTokens,
		"created_at":       c.CreatedAt,
		"updated_at":       c.UpdatedAt,
		"metadata":         c.Metadata,
	}
}
