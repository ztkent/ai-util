package aiutil

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strings"

	"github.com/PuerkitoBio/goquery"
)

// Determine if the user's input contains a resource command
// There is usually some limit to the number of tokens
func ManageResources(conv *Conversation, userInput string) (string, []string, error) {
	if conv == nil {
		return userInput, []string{}, fmt.Errorf("Failed to ManageResources: Conversation is nil")
	} else if len(userInput) == 0 {
		return userInput, []string{}, nil
	}

	// Only supporting URL and File resources for now
	var resourceCommands = []string{"url", "file"}
	resourcesFound := []string{}
	for _, cmd := range resourceCommands {
		re := regexp.MustCompile(fmt.Sprintf(`\-%s:(.*)`, cmd))
		matches := re.FindAllStringSubmatch(strings.ToLower(userInput), -1)
		for _, match := range matches {
			if len(match) > 1 {
				resource := strings.TrimSpace(match[1])
				resourcesFound = append(resourcesFound, cmd+":"+resource)
				err := AddResource(conv, resource, cmd)
				if err != nil {
					return userInput, resourcesFound, err
				}
				userInput = strings.Replace(userInput, "-"+cmd+":"+resource, "", -1)
			}
		}
	}
	return userInput, resourcesFound, nil
}

// Generate a resource message based on the path and type, return the message to append to the conversation
func AddResource(conv *Conversation, path string, pathType string) error {
	if pathType == "url" {
		err := AddURLReference(conv, path)
		if err != nil {
			return err
		}
	} else if pathType == "file" {
		err := AddFileMessage(conv, path)
		if err != nil {
			return err
		}
	} else {
		return fmt.Errorf("Invalid resource type: " + pathType)
	}
	return nil
}

func AddURLReference(conv *Conversation, path string) error {
	url, err := url.Parse(path)
	if err != nil {
		return err
	}

	if url.Scheme == "" || url.Host == "" {
		return fmt.Errorf("Invalid URL: " + path)
	}
	resp, err := http.Get(path)
	if err != nil {
		return fmt.Errorf("Failed to fetch URL: " + path)
	}
	defer resp.Body.Close()

	// Handle the page content
	doc, err := goquery.NewDocumentFromReader(resp.Body)
	if err != nil {
		return fmt.Errorf("Failed to parse HTML: " + path)
	}
	var pageContent []string
	seen := make(map[string]struct{})
	doc.Find("body").Each(func(i int, s *goquery.Selection) {
		text := strings.TrimSpace(s.Text())
		text = strings.ReplaceAll(text, "\n", "")
		if _, ok := seen[text]; text != "" && !ok {
			pageContent = append(pageContent, text)
			seen[text] = struct{}{}
		}
	})
	html := strings.Join(pageContent, " ")

	// Add the reference message
	if html != "" {
		return conv.AddReference(path, html)
	}
	return fmt.Errorf("Skipped adding empty URL reference: " + path)
}

func AddFileMessage(conv *Conversation, path string) error {
	resContent := ""
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		file, err := os.Open(path)
		if err != nil {
			return err
		}
		defer file.Close()
		fileInfo, err := file.Stat()
		if err != nil {
			return err
		}
		fileSize := fileInfo.Size()
		fileContents := make([]byte, fileSize)
		_, err = file.Read(fileContents)
		if err != nil {
			return err
		}

		resJson, err := json.Marshal(map[string]interface{}{
			"path":     path,
			"contents": string(fileContents),
		})
		if err != nil {
			return err
		}
		resContent = string(resJson)
	} else {
		return fmt.Errorf("Invalid file path: " + path)
	}

	// Add the reference message
	if resContent != "" {
		return conv.AddReference(path, resContent)
	}
	return fmt.Errorf("Skipped adding empty file reference: " + path)
}
