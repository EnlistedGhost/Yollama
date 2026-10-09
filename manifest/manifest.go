package manifest

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/EnlistedGhost/Yollama/types/model"
)

type Manifest struct {
	SchemaVersion int     `json:"schemaVersion"`
	MediaType     string  `json:"mediaType"`
	Config        Layer   `json:"config"`
	Layers        []Layer `json:"layers"`

	filepath string
	fi       os.FileInfo
	digest   string
}

func (m *Manifest) Size() (size int64) {
	for _, layer := range append(m.Layers, m.Config) {
		size += layer.Size
	}

	return
}

func (m *Manifest) Digest() string {
	return m.digest
}

func (m *Manifest) FileInfo() os.FileInfo {
	return m.fi
}

// ReadConfigJSON reads and unmarshals a config layer as JSON.
func (m *Manifest) ReadConfigJSON(configPath string, v any) error {
	for _, layer := range m.Layers {
		if layer.MediaType == "application/vnd.yollama.image.json" && layer.Name == configPath {
			blobPath, err := BlobsPath(layer.Digest)
			if err != nil {
				return err
			}
			data, err := os.ReadFile(blobPath)
			if err != nil {
				return err
			}
			return json.Unmarshal(data, v)
		}
	}
	return fmt.Errorf("config %q not found in manifest", configPath)
}

func (m *Manifest) Remove() error {
	if err := os.Remove(m.filepath); err != nil {
		return err
	}
	
	// Properly capture all 4 values returned by your updated Path() function
	primaryPath, auxPath, _, err := Path()
	if err != nil {
		return err
	}
	
	// Clean up primary empty folders
	if err := PruneDirectory(primaryPath); err != nil {
		return err
	}
	
	// Clean up auxiliary empty folders if configured
	if auxPath != "None" && auxPath != "" {
		_ = PruneDirectory(auxPath)
	}
	
	return nil
}

func (m *Manifest) RemoveLayers() error {
	ms, err := Manifests(true)
	if err != nil {
		return err
	}

	// Build set of digests still in use by other manifests
	inUse := make(map[string]struct{})
	for _, other := range ms {
		for _, layer := range append(other.Layers, other.Config) {
			if layer.Digest != "" {
				inUse[layer.Digest] = struct{}{}
			}
		}
	}

	// Remove layers not used by any other manifest
	for _, layer := range append(m.Layers, m.Config) {
		if layer.Digest == "" {
			continue
		}
		if _, used := inUse[layer.Digest]; used {
			continue
		}
		blob, err := BlobsPath(layer.Digest)
		if err != nil {
			return err
		}
		if err := os.Remove(blob); os.IsNotExist(err) {
			slog.Debug("layer does not exist", "digest", layer.Digest)
		} else if err != nil {
			return err
		}
	}

	return nil
}

func ParseNamedManifest(n model.Name) (*Manifest, error) {
	if !n.IsFullyQualified() {
		return nil, model.Unqualified(n)
	}

	// Use our new smart looker instead of forcing a primary path join
	p, err := PathForName(n)
	if err != nil {
		return nil, err
	}

	var m Manifest
	f, err := os.Open(p)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}

	sha256sum := sha256.New()
	if err := json.NewDecoder(io.TeeReader(f, sha256sum)).Decode(&m); err != nil {
		return nil, err
	}

	m.filepath = p
	m.fi = fi
	m.digest = hex.EncodeToString(sha256sum.Sum(nil))
	return &m, nil
}

func WriteManifest(name model.Name, config Layer, layers []Layer) error {
	// 1. Explicitly unpack all 4 values to satisfy your updated Path() signature
	primaryPath, auxPath, fAux, err := Path()
	if err != nil {
		return err
	}

	p := ""
	if fAux == true {
		p = filepath.Join(auxPath, name.Filepath())
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return err
		}
	} else {
		p = filepath.Join(primaryPath, name.Filepath())
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return err
		}
	}

	f, err := os.Create(p)
	if err != nil {
		return err
	}
	defer f.Close()

	m := Manifest{
		SchemaVersion: 2,
		MediaType:     "application/vnd.docker.distribution.manifest.v2+json",
		Config:        config,
		Layers:        layers,
	}

	return json.NewEncoder(f).Encode(m)
}


func Manifests(continueOnError bool) (map[model.Name]*Manifest, error) {
	primaryPath, auxPath, inclAux, err := Path()
	if err != nil {
		return nil, err
	}

	var targetDirs []string
	if primaryPath != "" && primaryPath != "None" {
		targetDirs = append(targetDirs, primaryPath)
	}
	if inclAux && auxPath != "" && auxPath != "None" {
		targetDirs = append(targetDirs, auxPath)
	}

	ms := make(map[model.Name]*Manifest)

	// Define a custom recursive function that follows symlinks
	var walkFn func(basePath, currentPath string) error
	walkFn = func(basePath, currentPath string) error {
		entries, err := os.ReadDir(currentPath)
		if err != nil {
			if continueOnError {
				return nil
			}
			return err
		}

		for _, entry := range entries {
			fullPath := filepath.Join(currentPath, entry.Name())
			
			// Use os.Stat to resolve the true target behind symlinks
			fi, err := os.Stat(fullPath)
			if err != nil {
				if continueOnError {
					continue
				}
				return err
			}

			if fi.IsDir() {
				// Recurse into subdirectories
				if err := walkFn(basePath, fullPath); err != nil && !continueOnError {
					return err
				}
			} else {
				// We found a manifest file! Compute the relative tracking path
				rel, err := filepath.Rel(basePath, fullPath)
				if err != nil {
					if continueOnError {
						continue
					}
					return err
				}

				n := model.ParseNameFromFilepath(rel)
				if !n.IsValid() {
					n = model.ParseName(strings.ReplaceAll(rel, string(filepath.Separator), "/"))
					if !n.IsValid() {
						if continueOnError {
							continue
						}
						return fmt.Errorf("invalid name resolution: %s", rel)
					}
				}

				m, err := ParseNamedManifest(n)
				if err != nil {
					if continueOnError {
						continue
					}
					return fmt.Errorf("%s %w", n, err)
				}

				ms[n] = m
			}
		}
		return nil
	}

	// Scan each target directory using our custom walker
	for _, baseManifestDir := range targetDirs {
		// Log out exactly what directories Yollama is checking to be 100% sure
		fmt.Printf("[YOLLAMA DEBUG] | Actively sweeping directory: %s\n", baseManifestDir)
		
		if err := walkFn(baseManifestDir, baseManifestDir); err != nil && !continueOnError {
			return nil, err
		}
	}

	return ms, nil
}
