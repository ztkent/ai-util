package aiutil

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// EmbeddingRequest describes an embeddings call.
type EmbeddingRequest struct {
	Model string
	Input []string
}

// Embedding is one vector result.
type Embedding struct {
	Index     int
	Embedding []float32
}

// EmbeddingResponse is the result of an embeddings call.
type EmbeddingResponse struct {
	Model      string
	Embeddings []Embedding
	Usage      Usage
}

// Embed returns embeddings for the given inputs. It uses the same client,
// base URL, and retry policy as chat calls.
func (c *Client) Embed(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	if req == nil || req.Model == "" {
		return nil, fmt.Errorf("embedding model is required")
	}
	if len(req.Input) == 0 {
		return nil, fmt.Errorf("at least one input is required")
	}

	body, err := json.Marshal(map[string]any{
		"model": req.Model,
		"input": req.Input,
	})
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}

	var out *EmbeddingResponse
	err = c.retry.do(ctx, func() error {
		var callErr error
		out, callErr = c.doEmbed(ctx, body)
		return callErr
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *Client) doEmbed(ctx context.Context, body []byte) (*EmbeddingResponse, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
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
		Model string `json:"model"`
		Data  []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
		Usage *wireUsage `json:"usage"`
		Error *wireError `json:"error"`
	}
	if err := json.Unmarshal(raw, &wr); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	if wr.Error != nil {
		return nil, apiErrorFromWire(wr.Error)
	}

	out := &EmbeddingResponse{Model: wr.Model, Usage: usageFromWire(wr.Usage)}
	for _, d := range wr.Data {
		out.Embeddings = append(out.Embeddings, Embedding{Index: d.Index, Embedding: d.Embedding})
	}
	return out, nil
}
