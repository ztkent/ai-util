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
	"github.com/sirupsen/logrus"
)

// Determine if the user's input contains a resource command
// If so, manage the resource and add the result to the conversation
// There is a LIMIT to the number of tokens, sometimes the resource is too large
func (conv *Conversation) ManageRAG(userInput string) string {
	if len(userInput) == 0 {
		return ""
	}

	var resourceCommands = []string{"url", "file"}
	resourcesFound := []string{}
	for _, cmd := range resourceCommands {
		re := regexp.MustCompile(fmt.Sprintf(`\-%s:(.*)`, cmd))
		matches := re.FindAllStringSubmatch(userInput, -1)
		for _, match := range matches {
			if len(match) > 1 {
				resource := strings.TrimSpace(match[1])
				resourcesFound = append(resourcesFound, cmd+":"+resource)
				conv.GenerateResource(resource, cmd)
				userInput = strings.Replace(userInput, "-"+cmd+":"+resource, "", -1)
			}
		}
	}
	if len(resourcesFound) > 0 {
		logger.WithFields(logrus.Fields{
			"resources": resourcesFound,
			"input":     userInput,
		}).Debug("Resources added to user input")
	}
	return userInput
}

func (conv *Conversation) GenerateResource(path string, pathType string) error {
	if pathType == "url" {
		msg, err := GenerateURLMessage(path)
		if err != nil {
			return err
		}
		conv.Append(msg)
	} else if pathType == "file" {
		msg, err := GenerateFileMessage(path)
		if err != nil {
			return err
		}
		conv.Append(msg)
	} else {
		return fmt.Errorf("Invalid resource type: " + pathType)
	}
	return nil
}

func GenerateURLMessage(path string) (openai.ChatCompletionMessage, error) {
	url, err := url.Parse(path)
	if err != nil {
		return openai.ChatCompletionMessage{}, err
	}

	messageParts := make([]openai.ChatMessagePart, 0)
	if url.Scheme != "" && url.Host != "" {
		logger.WithFields(logrus.Fields{
			"url": path,
		}).Debug("Downloading URL")

		resp, err := http.Get(path)
		if err != nil {
			return openai.ChatCompletionMessage{}, err
		}
		defer resp.Body.Close()

		// Handle the page content
		doc, err := goquery.NewDocumentFromReader(resp.Body)
		if err != nil {
			return openai.ChatCompletionMessage{}, err
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
		logger.WithFields(logrus.Fields{
			"path": path,
		}).Debug("Reading file from path")

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
