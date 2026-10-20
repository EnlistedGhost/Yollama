package server

import (
	"log"
	"strings"
	"slices"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"encoding/json"

	"github.com/EnlistedGhost/Yollama/types/model"
	"github.com/EnlistedGhost/Yollama/template"
	"github.com/EnlistedGhost/Yollama/manifest"
)

// Clean Yollama OCI Media Types
const (
	MediaTypeYollamaModel     = "application/vnd.yollama.image.model"
	MediaTypeYollamaProjector = "application/vnd.yollama.image.projector"
	MediaTypeYollamaTemplate  = "application/vnd.yollama.image.template"
	MediaTypeYollamaSystem    = "application/vnd.yollama.image.system"
	MediaTypeYollamaAdapter   = "application/vnd.yollama.image.adapter"
)

// NormalizeMediaType intercepts legacy media types and translates them into the Yollama schema in-memory
func NormalizeMediaType(mediaType string) string {
	// Clean up any spacing or casing anomalies
	mt := strings.TrimSpace(strings.ToLower(mediaType))

	switch mt {
	case "application/vnd.ollama.image.model", MediaTypeYollamaModel:
		return MediaTypeYollamaModel
	case "application/vnd.ollama.image.projector", MediaTypeYollamaProjector:
		return MediaTypeYollamaProjector
	case "application/vnd.ollama.image.template", MediaTypeYollamaTemplate:
		return MediaTypeYollamaTemplate
	case "application/vnd.ollama.image.system", MediaTypeYollamaSystem:
		return MediaTypeYollamaSystem
	case "application/vnd.ollama.image.adapter", MediaTypeYollamaAdapter:
		return MediaTypeYollamaAdapter
	default:
		return mediaType
	}
}

// ModelCapabilities tracks what features the underlying GGUF layers support
type ModelCapabilities struct {
	HasModel     bool
	HasProjector bool // For vision models
	HasTemplate  bool
}

// EvaluateCapabilities scans a manifest's layers and determines model features using normalized types
func EvaluateCapabilities(layers []struct{ MediaType string }) ModelCapabilities {
	var caps ModelCapabilities

	for _, layer := range layers {
		normalized := NormalizeMediaType(layer.MediaType)
		
		switch normalized {
		case MediaTypeYollamaModel:
			caps.HasModel = true
		case MediaTypeYollamaProjector:
			caps.HasProjector = true
		case MediaTypeYollamaTemplate:
			caps.HasTemplate = true
		}
	}

	if !caps.HasModel {
		log.Println("Warning: no capabilities found for model - model layer missing")
	}

	return caps
}

func appendCapability(capabilities []model.Capability, capability model.Capability) []model.Capability {
	if slices.Contains(capabilities, capability) {
		return capabilities
	}
	return append(capabilities, capability)
}

// Capabilities returns the capabilities that the model supports
func (m *Model) Capabilities() []model.Capability {

	capabilities := []model.Capability{}
	var modelArch string

	capabilities = m.configCapabilities(capabilities)
	capabilities, modelArch = m.ggufCapabilities(capabilities)
	capabilities = m.projectorCapabilities(capabilities)
	capabilities = m.templateCapabilities(capabilities)
	capabilities = m.parserCapabilities(capabilities)
	capabilities = m.modelFamilyCapabilities(capabilities)
	capabilities = m.filterUnsupportedCapabilities(capabilities, modelArch)

	if len(capabilities) == 0 {
		slog.Warn("unknown capabilities for model", "model", m.Name)
	}

	return capabilities
}

func (m *Model) configCapabilities(capabilities []model.Capability) []model.Capability {
	for _, c := range m.Config.Capabilities {
		capabilities = appendCapability(capabilities, model.Capability(c))
	}
	return capabilities
}

func (m *Model) ggufCapabilities(capabilities []model.Capability) ([]model.Capability, string) {
	if m.ModelPath == "" || !m.isGGUF() {
		return capabilities, ""
	}

	capabilities = chatTemplateCapabilities(capabilities, m.metadata.String("tokenizer.chat_template"))

	if m.metadata.Valid("pooling_type") {
		capabilities = appendCapability(capabilities, model.CapabilityEmbedding)
	} else {
		// If no embedding is specified, we assume the model supports completion.
		capabilities = appendCapability(capabilities, model.CapabilityCompletion)
	}
	if m.metadata.Valid("vision.block_count") {
		capabilities = appendCapability(capabilities, model.CapabilityVision)
	}
	if m.metadata.Valid("audio.block_count") {
		capabilities = appendCapability(capabilities, model.CapabilityAudio)
	}

	return capabilities, m.metadata.String("general.architecture")
}

func chatTemplateCapabilities(capabilities []model.Capability, chatTemplate string) []model.Capability {
	if chatTemplate == "" {
		return capabilities
	}

	if chatTemplateHasThinkingSupport(chatTemplate) {
		capabilities = appendCapability(capabilities, model.CapabilityThinking)
	}

	return capabilities
}

func chatTemplateHasThinkingSupport(chatTemplate string) bool {
	if strings.Contains(chatTemplate, "<think>") && strings.Contains(chatTemplate, "</think>") {
		return true
	}

	// Some Qwen/DeepSeek templates strip prior reasoning by splitting assistant
	// content at </think>; llama.cpp can still extract reasoning from them.
	return (strings.Contains(chatTemplate, "content.split('</think>')") ||
		strings.Contains(chatTemplate, `content.split("</think>")`)) &&
		!strings.Contains(chatTemplate, "reasoning_content") &&
		!strings.Contains(chatTemplate, "<SPECIAL_12>")
}

func hasMoreCapabilities(candidate, current []model.Capability) bool {
	return len(candidate) > len(current)
}

func (m *Model) projectorCapabilities(capabilities []model.Capability) []model.Capability {
	if len(m.ProjectorPaths) == 0 {
		return capabilities
	}

	capabilities = appendCapability(capabilities, model.CapabilityVision)
	for _, md := range m.projectorMetadata {
		if projectorHasAudio(md) && !projectorSuppressesAudioCapability(md) {
			capabilities = appendCapability(capabilities, model.CapabilityAudio)
		}
	}

	return capabilities
}

func (m *Model) templateCapabilities(capabilities []model.Capability) []model.Capability {
	return capabilities
}

func (m *Model) parserCapabilities(capabilities []model.Capability) []model.Capability {
	return capabilities
}

func (m *Model) modelFamilyCapabilities(capabilities []model.Capability) []model.Capability {
	isGptoss := slices.Contains([]string{"gptoss", "gpt-oss"}, m.Config.ModelFamily)
	if isGptoss {
		capabilities = appendCapability(capabilities, model.CapabilityThinking)
	}

	return capabilities
}

func (m *Model) filterUnsupportedCapabilities(capabilities []model.Capability, modelArch string) []model.Capability {
	if suppressAudioCapability(m, modelArch) {
		capabilities = slices.DeleteFunc(capabilities, func(c model.Capability) bool {
			return c == model.CapabilityAudio
		})
	}

	return capabilities
}

func suppressAudioCapability(m *Model, arch string) bool {
	// TODO remove function entirely
	return false
}

func projectorHasAudio(md ggufMetadata) bool {
	// read directly: Keys reports qualified keys, the accessors qualify theirs
	for _, key := range md.Keys() {
		if key == "has_audio_encoder" || strings.HasSuffix(key, ".has_audio_encoder") {
			if b, ok := md.KV[key].(bool); ok && b {
				return true
			}
		}
	}

	return false
}

func projectorSuppressesAudioCapability(md ggufMetadata) bool {
	switch md.String("vision.projector_type") {
	case "gemma3nv":
		return true
	}

	return false
}

// CheckCapabilities checks if the model has the specified capabilities returning an error describing
// any missing or unknown capabilities
func (m *Model) CheckCapabilities(want ...model.Capability) error {
	available := m.Capabilities()
	var errs []error

	// Map capabilities to their corresponding error
	capToErr := map[model.Capability]error{
		model.CapabilityCompletion: errCapabilityCompletion,
		model.CapabilityInsert:     errCapabilityInsert,
		model.CapabilityVision:     errCapabilityVision,
		model.CapabilityAudio:      errCapabilityAudio,
		model.CapabilityEmbedding:  errCapabilityEmbedding,
		model.CapabilityThinking:   errCapabilityThinking,
	}

	for _, cap := range want {
		err, ok := capToErr[cap]
		if !ok {
			slog.Error("unknown capability", "capability", cap)
			return fmt.Errorf("unknown capability: %s", cap)
		}

		if !slices.Contains(available, cap) {
			errs = append(errs, err)
		}
	}

	var err error
	if len(errs) > 0 {
		err = fmt.Errorf("%w %w", errCapabilities, errors.Join(errs...))
	}

	if slices.Contains(errs, errCapabilityThinking) {
		return fmt.Errorf("%w. Thinking support is depricated type in this model", err)
	}

	return err
}

func GetModelCapabilities(name string) (*Model, error) {
	slog.Info("[YOLLAMA] | GetModelCapabilities() - Function Called")
	n, mf, err := CheckForManifest(name)
	if err != nil {
		slog.Info("[YOLLAMA] | GetModelCapabilities() - Name resolution error")
		return nil, err
	}

	m := &Model{
		Name:      n.String(),
		ShortName: n.DisplayShortest(),
		Digest:    mf.Digest(),
		Template:  template.DefaultTemplate,
	}

	if mf.Config.Digest != "" {
		filename, err := manifest.BlobsPath(mf.Config.Digest)
		if err != nil {
			slog.Info("[YOLLAMA] | GetModelCapabilities() - blobs path error")
			return nil, err
		}

		configFile, err := os.Open(filename)
		if err != nil {
			slog.Info("[YOLLAMA] | GetModelCapabilities() - open file error")
			return nil, err
		}
		defer configFile.Close()

		if err := json.NewDecoder(configFile).Decode(&m.Config); err != nil {
			slog.Info("[YOLLAMA] | GetModelCapabilities() - json model file decode error")
			return nil, err
		}
	}

	modelHasPooling := false
	ggufChatTemplate := ""
	for _, layer := range mf.Layers {
		normalizedType := NormalizeMediaType(layer.MediaType)
		filename, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			slog.Info("[YOLLAMA] | GetModelCapabilities() - manifest blobs digest error")
			return nil, err
		}

		switch normalizedType {
		case "application/vnd.yollama.image.model":
			m.ModelPath = filename
			m.ParentModel = layer.From
			if m.isGGUF() {
				md, err := readGGUFMetadata(layer.Digest)
				if err != nil {
					slog.Error("couldn't read model metadata", "error", err)
					break
				}
				m.metadata = md
				ggufChatTemplate = md.String("tokenizer.chat_template")
				m.HasChatTemplate = ggufChatTemplate != ""
				modelHasPooling = md.Valid("pooling_type")
			}
		case "application/vnd.yollama.image.projector":
			m.ProjectorPaths = append(m.ProjectorPaths, filename)
			if md, err := readGGUFMetadata(layer.Digest); err != nil {
				slog.Error("couldn't read projector metadata", "error", err)
			} else {
				m.projectorMetadata = append(m.projectorMetadata, md)
			}
		case "application/vnd.yollama.image.prompt",
			"application/vnd.yollama.image.template":
			bts, err := os.ReadFile(filename)
			if err != nil {
				slog.Info("[YOLLAMA] | GetModelCapabilities() - file name error")
				return nil, err
			}

			m.Template, err = template.Parse(string(bts))
			if err != nil {
				slog.Info("[YOLLAMA] | GetModelCapabilities() - template parsing error")
				return nil, err
			}
		case "application/vnd.yollama.image.system":
			bts, err := os.ReadFile(filename)
			if err != nil {
				slog.Info("[YOLLAMA] | GetModelCapabilities() - yollama.image.system error")
				return nil, err
			}

			m.System = string(bts)
		case "application/vnd.yollama.image.params":
			params, err := os.Open(filename)
			if err != nil {
				slog.Info("[YOLLAMA] | GetModelCapabilities() - yollama.image.system error")
				return nil, err
			}
			defer params.Close()

			if err = json.NewDecoder(params).Decode(&m.Options); err != nil {
				slog.Info("[YOLLAMA] | GetModelCapabilities() - params json decoder error")
				return nil, err
			}
		case "application/vnd.yollama.image.messages":
			msgs, err := os.Open(filename)
			if err != nil {
				slog.Info("[YOLLAMA] | GetModelCapabilities() - yollama.image.messages error")
				return nil, err
			}
			defer msgs.Close()

			if err = json.NewDecoder(msgs).Decode(&m.Messages); err != nil {
				slog.Info("[YOLLAMA] | GetModelCapabilities() - messages json parsing error")
				return nil, err
			}
		case "application/vnd.yollama.image.license":
			bts, err := os.ReadFile(filename)
			if err != nil {
				slog.Info("[YOLLAMA] | GetModelCapabilities() - yollama.image.license error")
				return nil, err
			}
			m.License = append(m.License, string(bts))
		}
	}

	ggufCaps := chatTemplateCapabilities(nil, ggufChatTemplate)
	slog.Info("using GGUF chat_template", "model", m.Name, "chat_template_capabilities", ggufCaps)

	if m.ModelPath != "" && m.isGGUF() && !modelHasPooling && !m.HasChatTemplate {
		slog.Warn("model is missing tokenizer.chat_template, chat responses may be poorly formatted", "model", m.Name)
	}

	return m, nil
}