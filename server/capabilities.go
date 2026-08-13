package server

import (
	"log"
	"strings"
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
