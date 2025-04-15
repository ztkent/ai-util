package aiutil

import (
	"strings"
)

type OpenAIModel string
type ReplicateModel string

const (
	// OpenAI Models
	GPT35Turbo OpenAIModel = "gpt-3.5-turbo"
	GPT4       OpenAIModel = "gpt-4"
	GPT4Turbo  OpenAIModel = "gpt-4-turbo"
	GPT4O      OpenAIModel = "gpt-4o"
	GPT4OMini  OpenAIModel = "gpt-4o-mini"
	O1Preview  OpenAIModel = "o1-preview"
	O1Mini     OpenAIModel = "o1-mini"
	GPT41      OpenAIModel = "gpt-4.1"
	// Open-Source Models via Replicate
	MetaLlama38b          ReplicateModel = "meta/meta-llama-3-8b"
	MetaLlama370b         ReplicateModel = "meta/meta-llama-3-70b"
	MetaLlama38bInstruct  ReplicateModel = "meta/meta-llama-3-8b-instruct"
	MetaLlama370bInstruct ReplicateModel = "meta/meta-llama-3-70b-instruct"
	Mistral7B             ReplicateModel = "mistralai/mistral-7b-v0.1"
	Mistral7BInstruct     ReplicateModel = "mistralai/mistral-7b-instruct-v0.2"
	Mixtral8x7BInstruct   ReplicateModel = "mistralai/mixtral-8x7b-instruct-v0.1"
)

func (o OpenAIModel) String() string {
	return string(o)
}

func (r ReplicateModel) String() string {
	return string(r)
}

func IsSupportedOpenAIModel(name string) (OpenAIModel, bool) {
	switch strings.ToLower(name) {
	case GPT35Turbo.String(), "turbo35":
		return GPT35Turbo, true
	case GPT4.String(), "gpt4":
		return GPT4, true
	case GPT4Turbo.String(), "turbo":
		return GPT4Turbo, true
	case GPT4O.String(), "gpt4o":
		return GPT4O, true
	case GPT4OMini.String(), "gpt4o-mini":
		return GPT4OMini, true
	case O1Preview.String(), "o1-preview":
		return O1Preview, true
	case O1Mini.String(), "o1-mini":
		return O1Mini, true
	case GPT41.String(), "gpt4.1":
		return GPT41, true
	default:
		return "", false
	}
}

func IsSupportedReplicateModel(name string) (ReplicateModel, bool) {
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
