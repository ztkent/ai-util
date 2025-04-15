package aiutil

import (
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
)

const (
	// Limit resource size to avoid excessive token usage.
	// Resources that exceed this size will be truncated.
	MaxResourceContentLength = 50000
)

// AddResource is a helper to dispatch to specific resource adders.
func AddResource(conv *Conversation, path string, pathType string) error {
	switch strings.ToLower(pathType) {
	case "url":
		return AddURLReference(conv, path)
	case "file":
		return AddFileReference(conv, path) // Renamed from AddFileMessage
	default:
		return fmt.Errorf("unsupported resource type: %s", pathType)
	}
}

// AddURLReference fetches URL content (text) and adds it as a system reference message.
func AddURLReference(conv *Conversation, urlStr string) error {
	if conv == nil {
		return fmt.Errorf("conversation cannot be nil")
	}
	if !conv.ResourcesEnabled {
		return fmt.Errorf("resource management is disabled for this conversation")
	}

	parsedURL, err := url.ParseRequestURI(urlStr) // Stricter parsing
	if err != nil || (parsedURL.Scheme != "http" && parsedURL.Scheme != "https") {
		return fmt.Errorf("invalid or unsupported URL scheme: %s", urlStr)
	}

	// Consider adding a timeout to the HTTP client used here
	client := http.Client{Timeout: 15 * time.Second}
	resp, err := client.Get(parsedURL.String())
	if err != nil {
		return fmt.Errorf("failed to fetch URL %s: %w", urlStr, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("failed to fetch URL %s: status code %d", urlStr, resp.StatusCode)
	}

	// Limit reading to avoid huge downloads
	limitedReader := io.LimitReader(resp.Body, MaxResourceContentLength*2) // Read a bit more to check if truncated

	// Use goquery to extract text content
	doc, err := goquery.NewDocumentFromReader(limitedReader)
	if err != nil {
		return fmt.Errorf("failed to parse HTML from %s: %w", urlStr, err)
	}

	// Extract text, trying to be cleaner
	var pageText strings.Builder
	doc.Find("body").Each(func(i int, s *goquery.Selection) {
		cleanedText := strings.Join(strings.Fields(s.Text()), " ")
		pageText.WriteString(cleanedText)
		pageText.WriteString(" ")
	})

	content := strings.TrimSpace(pageText.String())

	// Truncate if necessary
	if len(content) > MaxResourceContentLength {
		content = content[:MaxResourceContentLength] + "..."
		fmt.Printf("Warning: Truncated content from URL %s to %d characters\n", urlStr, MaxResourceContentLength)
	}

	if content == "" {
		return fmt.Errorf("no text content extracted from URL: %s", urlStr)
	}

	// Add the reference message
	err = conv.AddReference(urlStr, content)
	if err != nil {
		return fmt.Errorf("failed to add URL reference %s to conversation: %w", urlStr, err)
	}

	return nil
}

// AddFileReference reads file content and adds it as a system reference message.
func AddFileReference(conv *Conversation, path string) error {
	if conv == nil {
		return fmt.Errorf("conversation cannot be nil")
	}
	if !conv.ResourcesEnabled {
		return fmt.Errorf("resource management is disabled for this conversation")
	}

	fileInfo, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("file not found: %s", path)
		}
		return fmt.Errorf("failed to stat file %s: %w", path, err)
	}

	if fileInfo.IsDir() {
		return fmt.Errorf("path is a directory, not a file: %s", path)
	}

	if fileInfo.Size() == 0 {
		return fmt.Errorf("file is empty: %s", path)
	}

	// Limit file size
	if fileInfo.Size() > MaxResourceContentLength {
		fmt.Printf("Warning: File %s (%d bytes) exceeds limit (%d bytes), truncating.\n", path, fileInfo.Size(), MaxResourceContentLength)
	}

	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open file %s: %w", path, err)
	}
	defer file.Close()

	// Read content with limit
	limitedReader := io.LimitReader(file, MaxResourceContentLength)
	contentBytes, err := io.ReadAll(limitedReader)
	if err != nil {
		return fmt.Errorf("failed to read file %s: %w", path, err)
	}

	content := string(contentBytes)
	if fileInfo.Size() > MaxResourceContentLength {
		content += "..."
	}

	// Add the reference message
	err = conv.AddReference(path, content)
	if err != nil {
		return fmt.Errorf("failed to add file reference %s to conversation: %w", path, err)
	}

	return nil
}
