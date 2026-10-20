package server

import (
	"fmt"
	"log/slog"

	"github.com/EnlistedGhost/Yollama/api"
	"github.com/EnlistedGhost/Yollama/fs/gguf"
	"github.com/EnlistedGhost/Yollama/manifest"
	"github.com/EnlistedGhost/Yollama/parser"
	"github.com/EnlistedGhost/Yollama/template"
	"github.com/EnlistedGhost/Yollama/types/model"
	
)

type Model struct {
	Name               string `json:"name"`
	Config             model.ConfigV2
	ShortName          string
	ModelPath          string
	ParentModel        string
	HasChatTemplate    bool
	ProjectorPaths     []string
	System             string
	License            []string
	Digest             string
	Options            map[string]any
	Messages           []api.Message

	Template *template.Template

	// Metadata of the model blob and of each projector, read from their
	// metadata files when the model is loaded.
	metadata          ggufMetadata
	projectorMetadata []ggufMetadata
}

func (m *Model) isGGUF() bool {
	return m.Config.ModelFormat == "" || m.Config.ModelFormat == "gguf"
}

func (m *Model) String() string {
	var modelfile parser.Modelfile

	modelfile.Commands = append(modelfile.Commands, parser.Command{
		Name: "model",
		Args: m.ModelPath,
	})

	for _, projector := range m.ProjectorPaths {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "model",
			Args: projector,
		})
	}

	if m.Template != nil {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "template",
			Args: m.Template.String(),
		})
	}

	if m.System != "" {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "system",
			Args: m.System,
		})
	}

	if m.Config.Renderer != "" {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "renderer",
			Args: m.Config.Renderer,
		})
	}

	if m.Config.Parser != "" {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "parser",
			Args: m.Config.Parser,
		})
	}

	for k, v := range m.Options {
		switch v := v.(type) {
		case []any:
			for _, s := range v {
				modelfile.Commands = append(modelfile.Commands, parser.Command{
					Name: k,
					Args: fmt.Sprintf("%v", s),
				})
			}
		default:
			modelfile.Commands = append(modelfile.Commands, parser.Command{
				Name: k,
				Args: fmt.Sprintf("%v", v),
			})
		}
	}

	for _, license := range m.License {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "license",
			Args: license,
		})
	}

	for _, msg := range m.Messages {
		modelfile.Commands = append(modelfile.Commands, parser.Command{
			Name: "message",
			Args: fmt.Sprintf("%s: %s", msg.Role, msg.Content),
		})
	}

	return modelfile.String()
}

func resolveModelManifest(name string) (model.Name, *manifest.Manifest, error) {
	n := model.ParseName(name)
	
	// If the name is a short name (unqualified), resolve it to its existing full path
	if !n.IsFullyQualified() {
		existing, err := getExistingName(n)
		if err != nil {
			return model.Name{}, nil, fmt.Errorf("failed to resolve unqualified name %q: %w", name, err)
		}
		n = existing
	}

	// Fetch the manifest cleanly using our dual-path system
	mf, err := manifest.ParseNamedManifest(n)
	if err != nil {
		return model.Name{}, nil, err
	}

	return n, mf, nil
}

func CheckForManifest(name string) (model.Name, *manifest.Manifest, error) {
	// Query local resolveModelManifest function
	n, mf, err := resolveModelManifest(name)
	if err != nil {
		return model.Name{}, nil, err
	}

	//Return to globally requested call
	return n, mf, nil
}

// Re-implemented Yollama's original in-house "CheckForModel" function
func CheckForModel(name string) (*Model, error) {
	slog.Info("[YOLLAMA] | CheckForModel() - Function Called")
	n, mf, err := resolveModelManifest(name)
	if err != nil {
		slog.Info("[YOLLAMA] | CheckForModel() - Name resolution error")
		return nil, err
	}

	m := &Model{
		Name:      n.String(),
		ShortName: n.DisplayShortest(),
		Digest:    mf.Digest(),
	}

	for _, layer := range mf.Layers {
		normalizedType := NormalizeMediaType(layer.MediaType)
		filename, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			slog.Info("[YOLLAMA] | CheckForModel() - manifest blobs digest error")
			return nil, err
		}

		if normalizedType == "application/vnd.yollama.image.model" {
			m.ModelPath = filename
			if m.isGGUF() {
				f, err := gguf.Open(filename)
				if err != nil {
					slog.Error("[YOLLAMA] | CheckForModel - couldn't open model file", "error", err)
					return nil, err
				}
				f.Close()
				break
			}
		}
	}

	return m, nil
}