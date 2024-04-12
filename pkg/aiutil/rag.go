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
	"github.com/sashabaranov/go-openai"
)

// Determine if the user's input contains a resource command
// There is usually some limit to the number of tokens
func ManageRAG(conv *Conversation, userInput string) (string, []string, error) {
	if conv == nil {
		return userInput, []string{}, fmt.Errorf("Failed to ManageRAG: Conversation is nil")
	} else if len(userInput) == 0 {
		return userInput, []string{}, nil
	}

	// Only supporting URL and File resources for now
	var resourceCommands = []string{"url", "file"}
	resourcesFound := []string{}
	for _, cmd := range resourceCommands {
		re := regexp.MustCompile(fmt.Sprintf(`\-%s:(.*)`, cmd))
		matches := re.FindAllStringSubmatch(userInput, -1)
		for _, match := range matches {
			if len(match) > 1 {
				resource := strings.TrimSpace(match[1])
				resourcesFound = append(resourcesFound, cmd+":"+resource)
				contextMsg, err := GenerateResource(resource, cmd)
				if err != nil {
					return userInput, resourcesFound, err
				}
				err = conv.Append(contextMsg)
				if err != nil {
					return userInput, resourcesFound, fmt.Errorf("Failed to append resource to conversation: " + err.Error())
				}
				userInput = strings.Replace(userInput, "-"+cmd+":"+resource, "", -1)
			}
		}
	}
	return userInput, resourcesFound, nil
}

// Generate a resource message based on the path and type, return the message to append to the conversation
func GenerateResource(path string, pathType string) (openai.ChatCompletionMessage, error) {
	if pathType == "url" {
		msg, err := GenerateURLMessage(path)
		if err != nil {
			return openai.ChatCompletionMessage{}, err
		}
		return msg, nil
	} else if pathType == "file" {
		msg, err := GenerateFileMessage(path)
		if err != nil {
			return openai.ChatCompletionMessage{}, err
		}
		return msg, nil
	} else {
		return openai.ChatCompletionMessage{}, fmt.Errorf("Invalid resource type: " + pathType)
	}
}

func GenerateURLMessage(path string) (openai.ChatCompletionMessage, error) {
	url, err := url.Parse(path)
	if err != nil {
		return openai.ChatCompletionMessage{}, err
	}

	messageParts := make([]openai.ChatMessagePart, 0)
	if url.Scheme != "" && url.Host != "" {
		resp, err := http.Get(path)
		if err != nil {
			return openai.ChatCompletionMessage{}, fmt.Errorf("Failed to fetch URL: " + path)
		}
		defer resp.Body.Close()

		// Handle the page content
		doc, err := goquery.NewDocumentFromReader(resp.Body)
		if err != nil {
			return openai.ChatCompletionMessage{}, fmt.Errorf("Failed to parse HTML: " + path)
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

		// Build the response message
		messageParts = append(messageParts, openai.ChatMessagePart{
			Type: openai.ChatMessagePartTypeText,
			Text: "<Path>" + path + "</Path>",
		})
		messageParts = append(messageParts, openai.ChatMessagePart{
			Type: openai.ChatMessagePartTypeText,
			Text: "<Content> " + html + " </Content>",
		})
	} else {
		return openai.ChatCompletionMessage{}, fmt.Errorf("Invalid URL: " + path)
	}

	return openai.ChatCompletionMessage{
		Name:         "URL",
		Role:         openai.ChatMessageRoleSystem,
		MultiContent: messageParts,
	}, nil
}

func GenerateFileMessage(path string) (openai.ChatCompletionMessage, error) {
	resContent := ""
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		file, err := os.Open(path)
		if err != nil {
			return openai.ChatCompletionMessage{}, err
		}
		defer file.Close()
		fileInfo, err := file.Stat()
		if err != nil {
			return openai.ChatCompletionMessage{}, err
		}
		fileSize := fileInfo.Size()
		fileContents := make([]byte, fileSize)
		_, err = file.Read(fileContents)
		if err != nil {
			return openai.ChatCompletionMessage{}, err
		}

		resJson, err := json.Marshal(map[string]interface{}{
			"path":     path,
			"contents": string(fileContents),
		})
		if err != nil {
			return openai.ChatCompletionMessage{}, err
		}
		resContent = string(resJson)
	} else {
		return openai.ChatCompletionMessage{}, fmt.Errorf("Invalid file path: " + path)
	}

	// Build the response message
	messageParts := make([]openai.ChatMessagePart, 0)
	messageParts = append(messageParts, openai.ChatMessagePart{
		Type: openai.ChatMessagePartTypeText,
		Text: "<Path>" + path + "</Path>",
	})
	messageParts = append(messageParts, openai.ChatMessagePart{
		Type: openai.ChatMessagePartTypeText,
		Text: "<Content>" + resContent + "</Content>",
	})

	return openai.ChatCompletionMessage{
		Name:         "FILE",
		Role:         openai.ChatMessageRoleSystem,
		MultiContent: messageParts,
	}, nil
}
