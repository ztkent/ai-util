package aiutil

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/ztkent/ai-util/types"
)

// RetryConfig holds configuration for the retry logic
type RetryConfig struct {
	MaxAttempts    int           // Maximum number of retry attempts (default: 5)
	BaseDelay      time.Duration // Initial delay for exponential backoff (default: 2s)
	MaxDelay       time.Duration // Maximum delay between retries (default: 30s)
	FallbackModels []string      // Models to try in order on quota errors (optional)
}

// DefaultRetryConfig returns the default retry configuration
func DefaultRetryConfig() *RetryConfig {
	return &RetryConfig{
		MaxAttempts: 5,
		BaseDelay:   2 * time.Second,
		MaxDelay:    30 * time.Second,
	}
}

// CompletionFunc is the function signature for AI completion calls
type CompletionFunc func(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error)

// WithRetry executes a completion request with smart retry logic
// - Parses rate limit errors for suggested retry delays
// - Uses exponential backoff for other transient errors
// - Skips retries for non-retryable errors (auth, invalid request)
// - Falls back through models in FallbackModels on quota errors (if provided)
func WithRetry(ctx context.Context, req *types.CompletionRequest, config *RetryConfig, fn CompletionFunc) (*types.CompletionResponse, error) {
	if config == nil {
		config = DefaultRetryConfig()
	}

	var lastErr error
	delay := config.BaseDelay
	fallbackIndex := 0
	maxAttempts := config.MaxAttempts
	if maxAttempts <= 0 {
		maxAttempts = 5
	}

	for attempt := 1; attempt <= maxAttempts; attempt++ {
		resp, err := fn(ctx, req)
		if err == nil {
			return resp, nil
		}

		lastErr = err

		// Check if error is non-retryable
		if !IsRetryableError(err) {
			slog.Error("Non-retryable error, aborting",
				"attempt", attempt,
				"error", err)
			return nil, err
		}

		// Check if error is quota exceeded
		if IsQuotaExceededError(err) {
			if len(config.FallbackModels) > 0 && fallbackIndex < len(config.FallbackModels) {
				slog.Warn("Quota exceeded, falling back to different model",
					"attempt", attempt,
					"model", req.Model,
					"fallback_model", config.FallbackModels[fallbackIndex],
					"max_attempts", maxAttempts,
					"error", err)
				req.Model = config.FallbackModels[fallbackIndex]
				fallbackIndex++
			} else {
				slog.Error("Quota exceeded and no more fallback models available",
					"attempt", attempt,
					"max_attempts", maxAttempts,
					"model", req.Model,
					"error", err)
				return nil, err
			}
		}

		if attempt < maxAttempts {
			// Check for rate limit with suggested retry time
			if suggestedDelay := ParseRateLimitDelay(err); suggestedDelay > 0 {
				delay = suggestedDelay + time.Second // Add buffer
				if !IsQuotaExceededError(err) {
					slog.Warn("Rate limited, using suggested delay",
						"attempt", attempt,
						"max_attempts", maxAttempts,
						"delay", delay,
						"error", err)
				}
			} else {
				slog.Warn("Operation failed, retrying with backoff",
					"attempt", attempt,
					"max_attempts", maxAttempts,
					"delay", delay,
					"error", err)
			}

			// Sleep with context awareness
			select {
			case <-ctx.Done():
				return nil, fmt.Errorf("retry cancelled: %w", ctx.Err())
			case <-time.After(delay):
			}

			// Exponential backoff with max cap (only if not rate limited)
			if ParseRateLimitDelay(err) == 0 {
				maxDelay := config.MaxDelay
				if maxDelay <= 0 {
					maxDelay = 30 * time.Second
				}
				delay = time.Duration(math.Min(
					float64(delay*2),
					float64(maxDelay),
				))
			}
		}
	}

	return nil, fmt.Errorf("operation failed after %d attempts: %w", maxAttempts, lastErr)
}

// ParseRateLimitDelay extracts the suggested retry delay from rate limit errors
// Looks for patterns like "Please retry in 34.42245165s"
func ParseRateLimitDelay(err error) time.Duration {
	if err == nil {
		return 0
	}

	errStr := err.Error()

	// Check if it's a rate limit error
	if !strings.Contains(errStr, "429") && !strings.Contains(errStr, "rate") && !strings.Contains(errStr, "quota") {
		return 0
	}

	// Try to extract "Please retry in Xs" or "retry in Xs"
	re := regexp.MustCompile(`[Rr]etry in (\d+\.?\d*)s`)
	matches := re.FindStringSubmatch(errStr)
	if len(matches) >= 2 {
		if seconds, parseErr := strconv.ParseFloat(matches[1], 64); parseErr == nil {
			return time.Duration(seconds * float64(time.Second))
		}
	}

	// Default rate limit delay if we can't parse the suggested time
	return 30 * time.Second
}

// IsRetryableError determines if an error is worth retrying
func IsRetryableError(err error) bool {
	if err == nil {
		return false
	}

	errStr := strings.ToLower(err.Error())

	// Non-retryable errors
	nonRetryable := []string{
		"invalid_api_key",
		"authentication",
		"unauthorized",
		"permission denied",
		"invalid request",
		"bad request",
		"not found",
	}

	for _, pattern := range nonRetryable {
		if strings.Contains(errStr, pattern) {
			return false
		}
	}

	// Retryable errors (rate limits, server errors, timeouts)
	retryable := []string{
		"429",
		"500",
		"502",
		"503",
		"504",
		"timeout",
		"deadline exceeded",
		"connection",
		"rate",
		"quota",
		"server_error",
		"temporarily unavailable",
	}

	for _, pattern := range retryable {
		if strings.Contains(errStr, pattern) {
			return true
		}
	}

	// Default to retrying unknown errors
	return true
}

// IsQuotaExceededError checks if an error indicates quota has been exceeded
func IsQuotaExceededError(err error) bool {
	if err == nil {
		return false
	}

	errStr := strings.ToLower(err.Error())
	return strings.Contains(errStr, "quota")
}
