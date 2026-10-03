package aiutil

import (
	"errors"
	"fmt"
	"time"
)

// APIError is returned when OpenRouter responds with a non-2xx status.
type APIError struct {
	StatusCode int           // HTTP status code
	Code       string        // provider error code, when present
	Message    string        // human-readable message
	Body       string        // raw response body, for debugging
	RetryAfter time.Duration // server-suggested wait, when provided
}

func (e *APIError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("openrouter: %d %s: %s", e.StatusCode, e.Code, e.Message)
	}
	return fmt.Sprintf("openrouter: %d: %s", e.StatusCode, e.Message)
}

// Retryable reports whether the request may succeed if retried.
func (e *APIError) Retryable() bool {
	if e.StatusCode == 429 || e.StatusCode >= 500 {
		return true
	}
	// Some upstream errors arrive without a status but with a code.
	switch e.Code {
	case "429", "rate_limit_exceeded", "server_error", "timeout":
		return true
	}
	return false
}

// IsAPIError reports whether err is an *APIError and returns it.
func IsAPIError(err error) (*APIError, bool) {
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr, true
	}
	return nil, false
}
