package aiutil

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"net"
	"time"
)

// RetryPolicy controls how failed requests are retried.
type RetryPolicy struct {
	MaxAttempts int           // total attempts, including the first
	BaseDelay   time.Duration // delay before the second attempt
	MaxDelay    time.Duration // upper bound on any single delay
}

// DefaultRetryPolicy retries transient failures with exponential backoff.
func DefaultRetryPolicy() RetryPolicy {
	return RetryPolicy{
		MaxAttempts: 4,
		BaseDelay:   500 * time.Millisecond,
		MaxDelay:    20 * time.Second,
	}
}

// do runs fn, retrying transient errors until the policy is exhausted or the
// context is cancelled. It honors a server-provided Retry-After when present.
// fn may return errStopRetry to abort immediately with the wrapped error.
func (p RetryPolicy) do(ctx context.Context, fn func() error) error {
	attempts := p.MaxAttempts
	if attempts < 1 {
		attempts = 1
	}
	delay := p.BaseDelay
	if delay <= 0 {
		delay = 500 * time.Millisecond
	}

	var lastErr error
	for attempt := 1; attempt <= attempts; attempt++ {
		lastErr = fn()
		if lastErr == nil {
			return nil
		}
		var stop *stopRetryError
		if errors.As(lastErr, &stop) {
			return stop.err
		}
		if attempt == attempts || !retryable(lastErr) {
			return lastErr
		}

		wait := delay
		if apiErr, ok := IsAPIError(lastErr); ok && apiErr.RetryAfter > 0 {
			wait = apiErr.RetryAfter
		}
		wait = jitter(wait)

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(wait):
		}

		delay = time.Duration(math.Min(float64(delay*2), float64(p.MaxDelay)))
	}
	return lastErr
}

// stopRetryError wraps an error to abort a retry loop immediately.
type stopRetryError struct{ err error }

func (e *stopRetryError) Error() string { return e.err.Error() }
func (e *stopRetryError) Unwrap() error { return e.err }

// stopRetry wraps err so a retry loop returns it without further attempts.
func stopRetry(err error) error { return &stopRetryError{err: err} }

// retryable reports whether an error is worth retrying.
func retryable(err error) bool {
	if err == nil {
		return false
	}
	if apiErr, ok := IsAPIError(err); ok {
		return apiErr.Retryable()
	}
	// Network-level failures (timeouts, resets, DNS) are transient.
	var netErr net.Error
	if errors.As(err, &netErr) {
		return true
	}
	return false
}

// jitter spreads retries to avoid synchronized thundering herds.
func jitter(d time.Duration) time.Duration {
	if d <= 0 {
		return 0
	}
	return d/2 + time.Duration(rand.Int63n(int64(d/2)+1))
}
