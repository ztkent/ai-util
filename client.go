package aiutil

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"time"
)

const defaultBaseURL = "https://openrouter.ai/api/v1"

// Request describes a single chat completion.
type Request struct {
	Model    string
	Messages []Message
	Tools    []Tool

	// ToolChoice controls tool use: "auto", "none", "required", or a specific
	// function. Nil lets the provider decide.
	ToolChoice any

	// Pointer fields distinguish "unset" from a meaningful zero value, so
	// Temperature: ptr(0) is a valid request.
	Temperature *float64
	TopP        *float64

	MaxTokens int
	Stop      []string

	// JSONMode asks the model to return a JSON object.
	JSONMode bool
}

// Response is the result of a chat completion.
type Response struct {
	ID           string
	Model        string
	Message      Message
	FinishReason string
	Usage        Usage
}

// Usage reports token counts for a request.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// Client talks to the OpenRouter API.
type Client struct {
	apiKey       string
	baseURL      string
	http         *http.Client
	retry        RetryPolicy
	defaultModel string
}

// Option configures a Client.
type Option func(*Client)

// New creates a Client. If apiKey is empty, OPENROUTER_API_KEY is used.
func New(apiKey string, opts ...Option) *Client {
	if apiKey == "" {
		apiKey = os.Getenv("OPENROUTER_API_KEY")
	}
	c := &Client{
		apiKey:  apiKey,
		baseURL: defaultBaseURL,
		http:    &http.Client{Timeout: 5 * time.Minute},
		retry:   DefaultRetryPolicy(),
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// WithBaseURL overrides the API base URL.
func WithBaseURL(url string) Option { return func(c *Client) { c.baseURL = url } }

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(hc *http.Client) Option { return func(c *Client) { c.http = hc } }

// WithDefaultModel sets the model used when a Request leaves Model empty.
func WithDefaultModel(model string) Option { return func(c *Client) { c.defaultModel = model } }

// WithRetry sets the retry policy.
func WithRetry(p RetryPolicy) Option { return func(c *Client) { c.retry = p } }

// Chat performs a completion and returns the full response.
func (c *Client) Chat(ctx context.Context, req *Request) (*Response, error) {
	wire, err := c.buildRequest(req, false)
	if err != nil {
		return nil, err
	}

	var resp *Response
	err = c.retry.do(ctx, func() error {
		var callErr error
		resp, callErr = c.doChat(ctx, wire)
		return callErr
	})
	if err != nil {
		return nil, err
	}
	return resp, nil
}

func (c *Client) doChat(ctx context.Context, wire *wireRequest) (*Response, error) {
	body, err := json.Marshal(wire)
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}

	httpResp, err := c.post(ctx, body)
	if err != nil {
		return nil, err
	}
	defer httpResp.Body.Close()

	raw, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return nil, parseAPIError(httpResp, raw)
	}

	var wr wireResponse
	if err := json.Unmarshal(raw, &wr); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	if wr.Error != nil {
		return nil, apiErrorFromWire(wr.Error)
	}
	if len(wr.Choices) == 0 {
		return nil, fmt.Errorf("openrouter: response contained no choices")
	}

	choice := wr.Choices[0]
	return &Response{
		ID:           wr.ID,
		Model:        wr.Model,
		Message:      fromWireMessage(choice.Message),
		FinishReason: choice.FinishReason,
		Usage:        usageFromWire(wr.Usage),
	}, nil
}

// buildRequest converts a Request into the wire format, applying defaults.
func (c *Client) buildRequest(req *Request, stream bool) (*wireRequest, error) {
	if req == nil {
		return nil, fmt.Errorf("request is nil")
	}
	model := req.Model
	if model == "" {
		model = c.defaultModel
	}
	if model == "" {
		return nil, fmt.Errorf("model is required")
	}
	if len(req.Messages) == 0 {
		return nil, fmt.Errorf("at least one message is required")
	}

	wire := &wireRequest{
		Model:       model,
		Messages:    toWireMessages(req.Messages),
		Temperature: req.Temperature,
		TopP:        req.TopP,
		MaxTokens:   req.MaxTokens,
		Stop:        req.Stop,
		Stream:      stream,
	}
	if len(req.Tools) > 0 {
		wire.Tools = toWireTools(req.Tools)
		wire.ToolChoice = req.ToolChoice
	}
	if req.JSONMode {
		wire.ResponseFormat = &responseFormat{Type: "json_object"}
	}
	if stream {
		wire.StreamOptions = &streamOptions{IncludeUsage: true}
	}
	return wire, nil
}

func (c *Client) post(ctx context.Context, body []byte) (*http.Response, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	httpReq.Header.Set("HTTP-Referer", "https://github.com/ztkent/ai-util")
	httpReq.Header.Set("X-Title", "ai-util")

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("send request: %w", err)
	}
	return resp, nil
}

func parseAPIError(resp *http.Response, body []byte) *APIError {
	apiErr := &APIError{StatusCode: resp.StatusCode, Body: string(body)}
	var parsed struct {
		Error *wireError `json:"error"`
	}
	if json.Unmarshal(body, &parsed) == nil && parsed.Error != nil {
		apiErr.Message = parsed.Error.Message
		if code, ok := parsed.Error.Code.(string); ok {
			apiErr.Code = code
		}
	}
	if apiErr.Message == "" {
		apiErr.Message = http.StatusText(resp.StatusCode)
	}
	if ra := resp.Header.Get("Retry-After"); ra != "" {
		if secs, err := strconv.Atoi(ra); err == nil {
			apiErr.RetryAfter = time.Duration(secs) * time.Second
		}
	}
	return apiErr
}

// apiErrorFromWire builds an APIError from an error embedded in a 200 response
// body. OpenRouter reports upstream failures this way, often with a numeric
// code that maps to an HTTP status.
func apiErrorFromWire(w *wireError) *APIError {
	apiErr := &APIError{Message: w.Message}
	switch code := w.Code.(type) {
	case string:
		apiErr.Code = code
	case float64:
		apiErr.StatusCode = int(code)
	}
	return apiErr
}

func usageFromWire(u *wireUsage) Usage {
	if u == nil {
		return Usage{}
	}
	return Usage{
		PromptTokens:     u.PromptTokens,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.TotalTokens,
	}
}
