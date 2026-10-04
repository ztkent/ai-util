package aiutil

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// ModelInfo describes a model available from the provider.
type ModelInfo struct {
	ID            string
	Name          string
	ContextLength int
	Pricing       ModelPricing
}

// ModelPricing is the per-token price in USD, as reported by the provider.
type ModelPricing struct {
	Prompt     float64 // USD per token
	Completion float64 // USD per token
}

// ListModels returns the models available from the provider. It is used to
// seed a local model catalog.
func (c *Client) ListModels(ctx context.Context) ([]ModelInfo, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/models", nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)

	httpResp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("send request: %w", err)
	}
	defer httpResp.Body.Close()

	raw, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return nil, parseAPIError(httpResp, raw)
	}

	var wr struct {
		Data []struct {
			ID            string `json:"id"`
			Name          string `json:"name"`
			ContextLength int    `json:"context_length"`
			Pricing       struct {
				Prompt     string `json:"prompt"`
				Completion string `json:"completion"`
			} `json:"pricing"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &wr); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	out := make([]ModelInfo, 0, len(wr.Data))
	for _, m := range wr.Data {
		out = append(out, ModelInfo{
			ID:            m.ID,
			Name:          m.Name,
			ContextLength: m.ContextLength,
			Pricing: ModelPricing{
				Prompt:     parsePrice(m.Pricing.Prompt),
				Completion: parsePrice(m.Pricing.Completion),
			},
		})
	}
	return out, nil
}

// parsePrice parses a provider price string (USD per token) into a float.
func parsePrice(s string) float64 {
	if s == "" {
		return 0
	}
	var f float64
	if _, err := fmt.Sscanf(s, "%g", &f); err != nil {
		return 0
	}
	return f
}
