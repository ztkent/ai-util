package aiutil

import (
	"fmt"
	"strings"
)

type OpenAIModel string
type ReplicateModel string
type AnyscaleModel string

const (
	// OpenAI Models
	GPT35Turbo OpenAIModel = "gpt-3.5-turbo" // IN: $0.50 / 1M tokens, OUT: $1.50 / 1M tokens
	GPT4       OpenAIModel = "gpt-4"         // IN: $30.00 / 1M tokens, OUT: $60.00 / 1M tokens
	GPT4Turbo  OpenAIModel = "gpt-4-turbo"   // IN: $10.00 / 1M tokens, OUT: $30.00 / 1M tokens
	// Open-Source Models via Replicate
	MetaLlama38b          ReplicateModel = "meta/meta-llama-3-8b"                 // IN: $0.05 / 1M tokens, OUT: $0.25 / 1M tokens
	MetaLlama370b         ReplicateModel = "meta/meta-llama-3-70b"                // IN: $0.65 / 1M tokens, OUT: $2.75 / 1M tokens
	MetaLlama38bInstruct  ReplicateModel = "meta/meta-llama-3-8b-instruct"        // IN: $0.05 / 1M tokens, OUT: $0.25 / 1M tokens
	MetaLlama370bInstruct ReplicateModel = "meta/meta-llama-3-70b-instruct"       // IN: $0.65 / 1M tokens, OUT: $2.75 / 1M tokens
	Mistral7B             ReplicateModel = "mistralai/mistral-7b-v0.1"            // IN: $0.05 / 1M tokens, OUT: $0.25 / 1M tokens
	Mistral7BInstruct     ReplicateModel = "mistralai/mistral-7b-instruct-v0.2"   // IN: $0.05 / 1M tokens, OUT: $0.25 / 1M tokens
	Mixtral8x7BInstruct   ReplicateModel = "mistralai/mixtral-8x7b-instruct-v0.1" // IN: $0.30 / 1M tokens, OUT: $1.00 / 1M tokens
	// Open-Source Models via Anyscale
	Anyscale_MetaLlama38bChat    AnyscaleModel = "meta-llama/Llama-3-8b-chat-hf"        // IN/OUT: $0.15 / 1M tokens
	Anyscale_MetaLlama370bChat   AnyscaleModel = "meta-llama/Llama-3-70b-chat-hf"       // IN/OUT: $1.00 / 1M tokens
	Anyscale_Mistral7BInstruct   AnyscaleModel = "mistralai/Mistral-7B-Instruct-v0.1"   // IN/OUT: $0.15 / 1M tokens
	Anyscale_Mixtral8x7BInstruct AnyscaleModel = "mistralai/Mixtral-8x7B-Instruct-v0.1" // IN/OUT: $0.50 / 1M tokens
	Anyscale_CodeLlama70b        AnyscaleModel = "codellama/CodeLlama-70b-Instruct-hf"  // IN/OUT: $1.00 / 1M tokens
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
	case GPT4.String(), "gpt4":
		return GPT4, true
	case GPT4Turbo.String(), "turbo":
		return GPT4Turbo, true
	default:
		return "", false
	}
}

func IsReplicateModel(name string) (ReplicateModel, bool) {
	switch strings.ToLower(name) {
	case MetaLlama38b.String(), "l3-8b":
		return MetaLlama38b, true
	case MetaLlama370b.String(), "l3-70b":
		return MetaLlama370b, true
	case MetaLlama38bInstruct.String(), "l3-8b-instruct":
		return MetaLlama38bInstruct, true
	case MetaLlama370bInstruct.String(), "l3-70b-instruct":
		return MetaLlama370bInstruct, true
	case Mistral7B.String(), "m7b":
		return Mistral7B, true
	case Mistral7BInstruct.String(), "m7b-instruct":
		return Mistral7BInstruct, true
	case Mixtral8x7BInstruct.String(), "m8x7b":
		return Mixtral8x7BInstruct, true
	default:
		return "", false
	}
}

func IsAnyscaleModel(name string) (AnyscaleModel, bool) {
	switch strings.ToLower(name) {
	case Anyscale_MetaLlama38bChat.String(), "l3-8b":
		return Anyscale_MetaLlama38bChat, true
	case Anyscale_MetaLlama370bChat.String(), "l3-70b":
		return Anyscale_MetaLlama370bChat, true
	case Anyscale_Mistral7BInstruct.String(), "m7b":
		return Anyscale_Mistral7BInstruct, true
	case Anyscale_Mixtral8x7BInstruct.String(), "m8x7b":
		return Anyscale_Mixtral8x7BInstruct, true
	case Anyscale_CodeLlama70b.String(), "cl70b":
		return Anyscale_CodeLlama70b, true
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
