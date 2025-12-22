package google

import (
	"context"
	"testing"

	"github.com/ztkent/ai-util/types"
)

func TestGoogleProvider_GetName(t *testing.T) {
	provider := NewProvider()
	if provider.GetName() != "google" {
		t.Errorf("Expected provider name 'google', got '%s'", provider.GetName())
	}
}

func TestGoogleProvider_ValidateModel(t *testing.T) {
	provider := NewProvider()

	// Test valid models
	validModels := []string{
		"gemini-2.5-pro",
		"gemini-2.5-flash",
		"gemini-3-pro-preview",
		"text-embedding-004",
	}

	for _, model := range validModels {
		if err := provider.ValidateModel(model); err != nil {
			t.Errorf("Expected model '%s' to be valid, got error: %v", model, err)
		}
	}

	// Test invalid model
	if err := provider.ValidateModel("invalid-model"); err == nil {
		t.Error("Expected invalid model to return error")
	}
}

func TestGoogleProvider_GetModels(t *testing.T) {
	provider := NewProvider()
	config := &Config{
		BaseConfig: types.BaseConfig{
			Provider: "google",
			APIKey:   "test-key",
		},
	}

	// Initialize provider
	if err := provider.Initialize(config); err != nil {
		t.Fatalf("Failed to initialize provider: %v", err)
	}

	// Get models
	ctx := context.Background()
	models, err := provider.GetModels(ctx)
	if err != nil {
		t.Fatalf("Failed to get models: %v", err)
	}

	if len(models) == 0 {
		t.Error("Expected models to be returned")
	}

	// Check that we have some expected models
	modelMap := make(map[string]*types.Model)
	for _, model := range models {
		modelMap[model.ID] = model
	}

	expectedModels := []string{
		"gemini-2.5-pro",
		"gemini-2.5-flash",
		"gemini-3-pro-preview",
	}

	for _, expectedModel := range expectedModels {
		if _, found := modelMap[expectedModel]; !found {
			t.Errorf("Expected model '%s' not found in models list", expectedModel)
		}
	}
}

func TestGoogleProvider_EstimateTokens(t *testing.T) {
	provider := NewProvider()

	messages := []*types.Message{
		{
			Role:     types.RoleUser,
			TextData: "Hello, how are you today?",
		},
		{
			Role:     types.RoleAssistant,
			TextData: "I'm doing well, thank you for asking!",
		},
	}

	ctx := context.Background()
	tokens, err := provider.EstimateTokens(ctx, messages, "gemini-2.5-flash")
	if err != nil {
		t.Fatalf("Failed to estimate tokens: %v", err)
	}

	if tokens <= 0 {
		t.Error("Expected positive token count")
	}
}

func TestConfig_Validate(t *testing.T) {
	// Test valid config
	config := &Config{
		BaseConfig: types.BaseConfig{
			Provider: "google",
			APIKey:   "test-key",
		},
	}

	if err := config.Validate(); err != nil {
		t.Errorf("Expected valid config, got error: %v", err)
	}

	// Test missing API key
	invalidConfig := &Config{
		BaseConfig: types.BaseConfig{
			Provider: "google",
		},
	}

	if err := invalidConfig.Validate(); err == nil {
		t.Error("Expected error for missing API key")
	}
}
