package types

import (
	"encoding/json"
	"fmt"
)

// Model represents a unified model across all providers
type Model struct {
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	Provider     string                 `json:"provider"`
	Description  string                 `json:"description,omitempty"`
	MaxTokens    int                    `json:"max_tokens,omitempty"`
	InputCost    float64                `json:"input_cost,omitempty"`   // Cost per 1M tokens
	OutputCost   float64                `json:"output_cost,omitempty"`  // Cost per 1M tokens
	Capabilities []string               `json:"capabilities,omitempty"` // e.g., "chat", "completion", "vision", "tools"
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// ModelCapability represents what a model can do
type ModelCapability string

const (
	CapabilityChat       ModelCapability = "chat"
	CapabilityCompletion ModelCapability = "completion"
	CapabilityVision     ModelCapability = "vision"
	CapabilityAudio      ModelCapability = "audio"
	CapabilityVideo      ModelCapability = "video"
	CapabilityTools      ModelCapability = "tools"
	CapabilityStreaming  ModelCapability = "streaming"
	CapabilityJSON       ModelCapability = "json"
	CapabilityThinking   ModelCapability = "thinking"
	CapabilityLive       ModelCapability = "live"
	CapabilityTTS        ModelCapability = "tts"
	CapabilityImage      ModelCapability = "image_generation"
)

// HasCapability checks if the model supports a specific capability
func (m *Model) HasCapability(capability ModelCapability) bool {
	for _, cap := range m.Capabilities {
		if cap == string(capability) {
			return true
		}
	}
	return false
}

// String returns a string representation of the model
func (m *Model) String() string {
	return fmt.Sprintf("%s/%s", m.Provider, m.ID)
}

// MarshalJSON implements custom JSON marshaling
func (m *Model) MarshalJSON() ([]byte, error) {
	type Alias Model
	return json.Marshal(&struct {
		*Alias
		FullName string `json:"full_name"`
	}{
		Alias:    (*Alias)(m),
		FullName: m.String(),
	})
}

// ModelRegistry manages available models across providers
type ModelRegistry struct {
	models map[string]*Model
}

// NewModelRegistry creates a new model registry
func NewModelRegistry() *ModelRegistry {
	return &ModelRegistry{
		models: make(map[string]*Model),
	}
}

// Register adds a model to the registry
func (r *ModelRegistry) Register(model *Model) {
	key := fmt.Sprintf("%s/%s", model.Provider, model.ID)
	r.models[key] = model
}

// Get retrieves a model by provider and ID
func (r *ModelRegistry) Get(provider, id string) (*Model, bool) {
	key := fmt.Sprintf("%s/%s", provider, id)
	model, exists := r.models[key]
	return model, exists
}

// GetByProvider returns all models for a specific provider
func (r *ModelRegistry) GetByProvider(provider string) []*Model {
	var models []*Model
	for _, model := range r.models {
		if model.Provider == provider {
			models = append(models, model)
		}
	}
	return models
}

// GetByCapability returns all models that support a specific capability
func (r *ModelRegistry) GetByCapability(capability ModelCapability) []*Model {
	var models []*Model
	for _, model := range r.models {
		if model.HasCapability(capability) {
			models = append(models, model)
		}
	}
	return models
}

// List returns all registered models
func (r *ModelRegistry) List() []*Model {
	var models []*Model
	for _, model := range r.models {
		models = append(models, model)
	}
	return models
}
