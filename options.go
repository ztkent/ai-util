package aiutil

import (
	"net/http"

	"github.com/replicate/replicate-go"
)

// ClientConfig holds the configuration for an AI client.
type ClientConfig struct {
	Provider    string
	Model       string
	APIKey      string
	BaseURL     string
	Temperature *float64
	TopP        *float64
	Seed        *int
	MaxTokens   *int // Max response tokens
	HTTPClient  *http.Client
	// OpenAI specific
	OrgID            string
	PresencePenalty  *float64
	FrequencyPenalty *float64
	ResponseFormat   string // e.g., "json_object"
	User             string
	// Replicate specific
	TopK           *int
	ReplicateInput map[string]interface{}
	Webhook        *replicate.Webhook
}

// Option defines the function signature for applying configuration options.
type Option func(*ClientConfig)

// WithProvider sets the AI provider.
func WithProvider(provider string) Option {
	return func(c *ClientConfig) {
		c.Provider = provider
	}
}

// WithModel sets the model identifier.
func WithModel(model string) Option {
	return func(c *ClientConfig) {
		c.Model = model
	}
}

// WithAPIKey sets the API key.
func WithAPIKey(apiKey string) Option {
	return func(c *ClientConfig) {
		c.APIKey = apiKey
	}
}

// WithBaseURL sets a custom base URL for the API endpoint.
func WithBaseURL(baseURL string) Option {
	return func(c *ClientConfig) {
		c.BaseURL = baseURL
	}
}

// WithTemperature sets the sampling temperature.
func WithTemperature(temp float64) Option {
	return func(c *ClientConfig) {
		c.Temperature = &temp
	}
}

// WithTopP sets the nucleus sampling parameter.
func WithTopP(topP float64) Option {
	return func(c *ClientConfig) {
		c.TopP = &topP
	}
}

// WithSeed sets the seed for deterministic outputs (if supported).
func WithSeed(seed int) Option {
	return func(c *ClientConfig) {
		c.Seed = &seed
	}
}

// WithMaxTokens sets the maximum number of tokens to generate in the response.
func WithMaxTokens(maxTokens int) Option {
	return func(c *ClientConfig) {
		c.MaxTokens = &maxTokens
	}
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(c *ClientConfig) {
		c.HTTPClient = client
	}
}

// WithOpenAIOrganizationID sets the OpenAI organization ID.
func WithOpenAIOrganizationID(orgID string) Option {
	return func(c *ClientConfig) {
		c.OrgID = orgID
	}
}

// WithOpenAIPresencePenalty sets the OpenAI presence penalty.
func WithOpenAIPresencePenalty(penalty float64) Option {
	return func(c *ClientConfig) {
		c.PresencePenalty = &penalty
	}
}

// WithOpenAIFrequencyPenalty sets the OpenAI frequency penalty.
func WithOpenAIFrequencyPenalty(penalty float64) Option {
	return func(c *ClientConfig) {
		c.FrequencyPenalty = &penalty
	}
}

// WithOpenAIResponseFormat sets the OpenAI response format (e.g., "json_object").
func WithOpenAIResponseFormat(format string) Option {
	return func(c *ClientConfig) {
		c.ResponseFormat = format
	}
}

// WithOpenAIUser sets the OpenAI user identifier.
func WithOpenAIUser(user string) Option {
	return func(c *ClientConfig) {
		c.User = user
	}
}

// WithReplicateTopK sets the Replicate top-k sampling parameter.
func WithReplicateTopK(topK int) Option {
	return func(c *ClientConfig) {
		c.TopK = &topK
	}
}

// WithReplicateInput sets arbitrary input parameters for Replicate models.
// Multiple calls will merge the input maps.
func WithReplicateInput(input map[string]interface{}) Option {
	return func(c *ClientConfig) {
		if c.ReplicateInput == nil {
			c.ReplicateInput = make(map[string]interface{})
		}
		for k, v := range input {
			c.ReplicateInput[k] = v
		}
	}
}

// WithReplicateWebhook configures a webhook for Replicate predictions.
func WithReplicateWebhook(url string, events []replicate.WebhookEventType) Option {
	return func(c *ClientConfig) {
		if url != "" && len(events) > 0 {
			c.Webhook = &replicate.Webhook{
				URL:    url,
				Events: events,
			}
		} else {
			c.Webhook = nil
		}
	}
}
