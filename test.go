package aiutil

import (
	"testing"

	"github.com/ztkent/ai-util/types"
)

func TestBuilder(t *testing.T) {
	// Test builder without actual API keys
	builder := NewAIClient().
		WithDefaultProvider("openai").
		WithDefaultModel("gpt-3.5-turbo").
		WithDefaultMaxTokens(1000).
		WithDefaultTemperature(0.5)

	// Test that builder configuration is set correctly
	if builder.config.DefaultProvider != "openai" {
		t.Errorf("Expected default provider to be 'openai', got %s", builder.config.DefaultProvider)
	}

	if builder.config.DefaultModel != "gpt-3.5-turbo" {
		t.Errorf("Expected default model to be 'gpt-3.5-turbo', got %s", builder.config.DefaultModel)
	}

	if builder.config.DefaultMaxTokens != 1000 {
		t.Errorf("Expected default max tokens to be 1000, got %d", builder.config.DefaultMaxTokens)
	}

	if builder.config.DefaultTemperature != 0.5 {
		t.Errorf("Expected default temperature to be 0.5, got %f", builder.config.DefaultTemperature)
	}
}

func TestMessage(t *testing.T) {
	// Test text message creation
	msg := types.NewTextMessage(types.RoleUser, "Hello, world!")

	if msg.Role != types.RoleUser {
		t.Errorf("Expected role to be 'user', got %s", msg.Role)
	}

	if msg.GetText() != "Hello, world!" {
		t.Errorf("Expected text to be 'Hello, world!', got %s", msg.GetText())
	}

	// Test content message creation
	content := []types.MessageContent{
		types.TextContent{Text: "What's in this image?"},
		types.ImageContent{URL: "https://example.com/image.jpg"},
	}

	contentMsg := types.NewContentMessage(types.RoleUser, content)

	if contentMsg.Role != types.RoleUser {
		t.Errorf("Expected role to be 'user', got %s", contentMsg.Role)
	}

	if len(contentMsg.Content) != 2 {
		t.Errorf("Expected 2 content items, got %d", len(contentMsg.Content))
	}

	if !contentMsg.HasImages() {
		t.Error("Expected message to have images")
	}
}

func TestModel(t *testing.T) {
	model := &types.Model{
		ID:           "gpt-3.5-turbo",
		Name:         "GPT-3.5 Turbo",
		Provider:     "openai",
		MaxTokens:    4096,
		Capabilities: []string{"chat", "streaming"},
	}

	if !model.HasCapability(types.CapabilityChat) {
		t.Error("Expected model to have chat capability")
	}

	if !model.HasCapability(types.CapabilityStreaming) {
		t.Error("Expected model to have streaming capability")
	}

	if model.HasCapability(types.CapabilityVision) {
		t.Error("Expected model to not have vision capability")
	}

	if model.String() != "openai/gpt-3.5-turbo" {
		t.Errorf("Expected model string to be 'openai/gpt-3.5-turbo', got %s", model.String())
	}
}

func TestModelRegistry(t *testing.T) {
	registry := types.NewModelRegistry()

	model1 := &types.Model{
		ID:       "gpt-3.5-turbo",
		Provider: "openai",
	}

	model2 := &types.Model{
		ID:       "meta-llama-3-8b-instruct",
		Provider: "replicate",
	}

	registry.Register(model1)
	registry.Register(model2)

	// Test retrieval
	retrieved, exists := registry.Get("openai", "gpt-3.5-turbo")
	if !exists {
		t.Error("Expected to find registered model")
	}

	if retrieved.ID != "gpt-3.5-turbo" {
		t.Errorf("Expected retrieved model ID to be 'gpt-3.5-turbo', got %s", retrieved.ID)
	}

	// Test provider filtering
	openaiModels := registry.GetByProvider("openai")
	if len(openaiModels) != 1 {
		t.Errorf("Expected 1 OpenAI model, got %d", len(openaiModels))
	}

	// Test listing all
	allModels := registry.List()
	if len(allModels) != 2 {
		t.Errorf("Expected 2 total models, got %d", len(allModels))
	}
}

func TestError(t *testing.T) {
	err := types.NewError(types.ErrCodeAuthentication, "Invalid API key", "openai")

	expectedMsg := "[openai] AUTHENTICATION_FAILED: Invalid API key"
	if err.Error() != expectedMsg {
		t.Errorf("Expected error message to be '%s', got '%s'", expectedMsg, err.Error())
	}

	if err.Code != types.ErrCodeAuthentication {
		t.Errorf("Expected error code to be %s, got %s", types.ErrCodeAuthentication, err.Code)
	}

	if err.Provider != "openai" {
		t.Errorf("Expected provider to be 'openai', got %s", err.Provider)
	}
}

func TestConversationConfig(t *testing.T) {
	// Test that we can create a client without actual API keys for testing
	client := NewClient(&ClientConfig{
		DefaultMaxTokens:   2048,
		DefaultTemperature: 0.8,
		ProviderConfigs:    make(map[string]types.Config),
	})

	config := &ConversationConfig{
		SystemPrompt:   "You are a test assistant",
		MaxTokens:      1024,
		AutoTruncate:   true,
		PreserveSystem: true,
	}

	conv := client.NewConversation(config)

	if conv.ID == "" {
		t.Error("Expected conversation to have an ID")
	}

	if conv.MaxTokens != 1024 {
		t.Errorf("Expected max tokens to be 1024, got %d", conv.MaxTokens)
	}

	if len(conv.GetMessages()) != 1 {
		t.Errorf("Expected 1 message (system), got %d", len(conv.GetMessages()))
	}

	systemMsg := conv.GetMessages()[0]
	if systemMsg.Role != types.RoleSystem {
		t.Errorf("Expected first message to be system, got %s", systemMsg.Role)
	}

	if systemMsg.GetText() != "You are a test assistant" {
		t.Errorf("Expected system message text to be 'You are a test assistant', got %s", systemMsg.GetText())
	}
}

func TestConversationMessages(t *testing.T) {
	client := NewClient(nil)
	conv := client.NewConversation(nil)

	// Test adding messages
	err := conv.AddUserMessage("Hello")
	if err != nil {
		t.Errorf("Unexpected error adding user message: %v", err)
	}

	err = conv.AddAssistantMessage("Hi there!")
	if err != nil {
		t.Errorf("Unexpected error adding assistant message: %v", err)
	}

	messages := conv.GetMessages()
	if len(messages) != 2 {
		t.Errorf("Expected 2 messages, got %d", len(messages))
	}

	// Test message filtering
	userMessages := conv.GetMessagesByRole(types.RoleUser)
	if len(userMessages) != 1 {
		t.Errorf("Expected 1 user message, got %d", len(userMessages))
	}

	assistantMessages := conv.GetMessagesByRole(types.RoleAssistant)
	if len(assistantMessages) != 1 {
		t.Errorf("Expected 1 assistant message, got %d", len(assistantMessages))
	}

	// Test last message
	lastMsg := conv.GetLastMessage()
	if lastMsg.Role != types.RoleAssistant {
		t.Errorf("Expected last message to be assistant, got %s", lastMsg.Role)
	}

	// Test clear
	conv.Clear()
	if len(conv.GetMessages()) != 0 {
		t.Errorf("Expected 0 messages after clear, got %d", len(conv.GetMessages()))
	}
}
