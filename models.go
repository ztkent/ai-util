package aiutil

import (
	"fmt"
	"strings"
)

type OpenAIModel string
type ReplicateModel string
type AnyscaleModel string

const (
	GPT35Turbo          OpenAIModel    = "gpt-3.5-turbo"
	GPT4TurboPreview    OpenAIModel    = "gpt-4-turbo-preview"
	GPT4Turbo           OpenAIModel    = "gpt-4-turbo"
	Llama38b            ReplicateModel = "meta-llama-3-8b"
	Llama38bInstruct    ReplicateModel = "meta-llama-3-8b-instruct"
	Llama370b           ReplicateModel = "meta-llama-3-70b"
	Llama370bInstruct   ReplicateModel = "meta-llama-3-70b-instruct"
	Mistral7BInstruct   AnyscaleModel  = "mistralai/Mistral-7B-Instruct-v0.1"
	Mixtral8x7BInstruct AnyscaleModel  = "mistralai/Mixtral-8x7B-Instruct-v0.1"
	Llama27bChat        AnyscaleModel  = "meta-llama/Llama-2-7b-chat-hf"
	Llama213bChat       AnyscaleModel  = "meta-llama/Llama-2-13b-chat-hf"
	Llama270bChat       AnyscaleModel  = "meta-llama/Llama-2-70b-chat-hf"
	CodeLlama34b        AnyscaleModel  = "codellama/CodeLlama-34b-Instruct-hf"
	CodeLlama70b        AnyscaleModel  = "codellama/CodeLlama-70b-Instruct-hf"
)

func (o OpenAIModel) String() string {
	return string(o)
}

func (r ReplicateModel) String() string {
	return string(r)
}

func (a AnyscaleModel) String() string {
	return string(a)
}

func IsOpenAIModel(name string) (OpenAIModel, bool) {
	switch strings.ToLower(name) {
	case GPT35Turbo.String(), "turbo35":
		return GPT35Turbo, true
	case GPT4TurboPreview.String(), "turbopreview":
		return GPT4TurboPreview, true
	case GPT4Turbo.String(), "turbo":
		return GPT4Turbo, true
	default:
		return "", false
	}
}

func IsReplicateModel(name string) (ReplicateModel, bool) {
	switch strings.ToLower(name) {
	case Llama38b.String(), "l3-8b":
		return Llama38b, true
	case Llama38bInstruct.String(), "l3-8b-instruct":
		return Llama38bInstruct, true
	case Llama370b.String(), "l3-70b":
		return Llama370b, true
	case Llama370bInstruct.String(), "l3-70b-instruct":
		return Llama370bInstruct, true
	default:
		return "", false
	}
}

func IsAnyscaleModel(name string) (AnyscaleModel, bool) {
	switch strings.ToLower(name) {
	case Mistral7BInstruct.String(), "m7b":
		return Mistral7BInstruct, true
	case Mixtral8x7BInstruct.String(), "m8x7b":
		return Mixtral8x7BInstruct, true
	case Llama27bChat.String(), "l2-7b":
		return Llama27bChat, true
	case Llama213bChat.String(), "l2-13b":
		return Llama213bChat, true
	case Llama270bChat.String(), "l2-70b":
		return Llama270bChat, true
	case CodeLlama34b.String(), "cl34b":
		return CodeLlama34b, true
	case CodeLlama70b.String(), "cl70b":
		return CodeLlama70b, true
	default:
		return "", false
	}
}

func GetModelProvider(name string) (Provider, error) {
	if _, ok := IsOpenAIModel(name); ok {
		return OpenAI, nil
	} else if _, ok := IsReplicateModel(name); ok {
		return Replicate, nil
	} else if _, ok := IsAnyscaleModel(name); ok {
		return Anyscale, nil
	}
	return "", fmt.Errorf("Invalid model name: %s", name)
}
