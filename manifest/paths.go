package manifest

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/EnlistedGhost/Yollama/envconfig"
	"github.com/EnlistedGhost/Yollama/types/model"
)

var ErrInvalidDigestFormat = errors.New("invalid digest format")

// getYollamaTLSConfigPath returns the full path to the llama-cpp_v.conf file.
func get_Aux_Model_Path() (string, error) {
    // 1. Get the dynamic home directory path
    rootDir, err := os.UserHomeDir()
    if err != nil {
        fmt.Errorf("[YOLLAMA] | Aux model (secondary) pathing file error: %v", err)
        return "", err
    }

    // 2. Safely join the home directory with the .yollama folder
    v_file_path := filepath.Join(rootDir, ".yollama")
    fmt.Println("Aux model Config - Checking Primary Directory:", v_file_path)

    v_Llama_Path := filepath.Join(v_file_path, "yollama_paths.conf")
    return v_Llama_Path, err
}

// readYollamaTLSConfig reads the config file and returns true if TLS is enabled.
// The file should contain either 1 (true) or 0 (false).
func read_Aux_Model_Path(found_model_aux_path string) (string, error) {
    // Read entire file into byte slice
    get_Aux_Model_Path, err := os.ReadFile(found_model_aux_path)
    if err != nil {
        return "None", fmt.Errorf("[YOLLAMA] | failed to read aux model config file: %w", err)
    }

    // Convert bytes to string (trim whitespace and newlines)
    path_string := strings.TrimSpace(string(get_Aux_Model_Path))

    return path_string, nil
}

func report_Aux_Model_Path() (string, error) {
	// Get the aux model config file path
	path_aux_models, err := get_Aux_Model_Path()
    if err != nil {
        fmt.Errorf("[YOLLAMA] | ❌ Failed to resolve aux model-location config file path.", "error", err)
		return "None", err
    } else {
        fmt.Printf("[YOLLAMA] | ✅ Fetched configured aux model-location file path: %s\n", path_aux_models)
    }

    secondary_aux_model_path, err := read_Aux_Model_Path(path_aux_models)
    if err != nil {
        // Fail open to "None"
        fmt.Printf("[YOLLAMA] | ⚠️ Llama.cpp aux model-location file unreadable; defaulting to 'None'", "error", err)
        return "None", err
    } else {
      	fmt.Printf("[YOLLAMA] | ✅ Fetched aux model-location file with value: %s\n", secondary_aux_model_path)
    }

    return secondary_aux_model_path, nil
}

func report_Primary_Model_Path() (string, error) {
	path := filepath.Join(envconfig.Models(), "manifests")
	if err := os.MkdirAll(path, 0o755); err != nil {
		return "None", fmt.Errorf("%w: ensure path elements are traversable", err)
	}

	return path, nil
}

func PrimaryPath() (string, error) {
	Pime_Model_Path, err := report_Primary_Model_Path()
	if err != nil {
		return "None", fmt.Errorf("Primary file path Check Failure!")
	}

	return Pime_Model_Path, nil
}

func AuxPath() (string, error) {
	Aux_Model_Path, err_path := report_Aux_Model_Path()
	if err_path != nil {
		return "None", fmt.Errorf("Aux file path Check Failure!")
	}

	path := filepath.Join(Aux_Model_Path, "manifests")
	if err := os.MkdirAll(path, 0o755); err != nil {
		return "None", fmt.Errorf("%w: ensure path elements are traversable", err)
	}

	return path, nil
}

func Path() (string, string, bool, error) {
	path, err := PrimaryPath()
	if err != nil {
		fmt.Printf("[YOLLAMA] | ⚠️ Primary Model-Storage location and pathing disabled!")
	}

	path_aux, aux_err := AuxPath()
	incl_aux := true
	if aux_err != nil {
		incl_aux = false
		fmt.Printf("[YOLLAMA] | ⚠️ Auxiliary Model-Storage location and pathing disabled!")
	}

	if err != nil && aux_err != nil {
		return "None", path_aux, incl_aux, fmt.Errorf("%w: ensure path elements are traversable", err)
	}

	return path, path_aux, incl_aux, nil
}

// ScanAllManifests returns a combined list of manifests
func ScanAllManifests() ([]string, error) {
	primaryPath, auxPath, inclAux, err := Path()
	if err != nil {
		return nil, err
	}

	// Use a map to effortlessly prevent duplicate entries if a model is in both directories
	foundModels := make(map[string]bool)

	// Helper closure to scan a specific manifest folder
	scanDir := func(basePath string) {
		if basePath == "" || basePath == "None" {
			return
		}
		// Recursively walk the directory looking for manifest files
		_ = filepath.WalkDir(basePath, func(path string, d os.DirEntry, walkErr error) error {
			if walkErr != nil || d.IsDir() {
				return nil
			}
			// Get the relative path starting from the manifests folder root
			relPath, err := filepath.Rel(basePath, path)
			if err == nil {
				foundModels[relPath] = true
			}
			return nil
		})
	}

	// 1. Scan primary directory
	scanDir(primaryPath)

	// 2. Scan auxiliary directory if healthy
	if inclAux {
		scanDir(auxPath)
	}

	// Convert map keys back into a clean string slice of model paths
	var allModels []string
	for modelPath := range foundModels {
		allModels = append(allModels, modelPath)
	}

	return allModels, nil
}


// PathForName returns the path to the manifest file for a specific model name.
func PathForName(n model.Name) (string, error) {
	if !n.IsValid() {
		return "", os.ErrNotExist
	}

	model_manifest := "None"

	manifests, aux_manifests, _, err := Path()
	if err != nil {
		return "", err
	}

	primary_location:= filepath.Join(manifests, n.Filepath())
	auxiliary_location := filepath.Join(aux_manifests, n.Filepath())

	_, check_err_1 := os.ReadFile(primary_location)
	_, check_err_2 := os.ReadFile(auxiliary_location)

	if check_err_1 != nil && check_err_2 != nil {
		return "", err
	} else if check_err_1 != nil {
		model_manifest = auxiliary_location
	} else {
		model_manifest = primary_location
	}

	return model_manifest, nil
}

func BlobsPath(digest string) (string, error) {
	digest = strings.ReplaceAll(digest, ":", "-")

	// Get our primary and auxiliary base model paths via Path()
	manifestsDir, auxManifestsDir, inclAux, err := Path()
	if err != nil {
		return "", err
	}

	// Reconstruct the base models directory from the manifest directories
	// (Moving up one level from ".../models/manifests" to ".../models")
	primaryBase := filepath.Dir(manifestsDir)
	primaryBlobPath := filepath.Join(primaryBase, "blobs", digest)

	// If auxiliary pathing is healthy, check if the blob lives there first
	if inclAux && auxManifestsDir != "None" {
		auxBase := filepath.Dir(auxManifestsDir)
		auxBlobPath := filepath.Join(auxBase, "blobs", digest)

		// Check if the blob file actually exists in the auxiliary path
		if _, err := os.Lstat(auxBlobPath); err == nil {
			return auxBlobPath, nil
		}
	}

	// Fallback/Default to Primary path if not found in Aux
	dirPath := filepath.Dir(primaryBlobPath)
	if digest == "" {
		dirPath = primaryBlobPath
	}

	if err := os.MkdirAll(dirPath, 0o755); err != nil {
		return "", fmt.Errorf("%w: ensure path elements are traversable", err)
	}

	return primaryBlobPath, nil
}


// PruneDirectory removes empty directories recursively.
func PruneDirectory(path string) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}

	if info.IsDir() && info.Mode()&os.ModeSymlink == 0 {
		entries, err := os.ReadDir(path)
		if err != nil {
			return err
		}

		for _, entry := range entries {
			err := PruneDirectory(filepath.Join(path, entry.Name()))
			if err != nil {
				return err
			}
		}

		entries, err = os.ReadDir(path)
		if err != nil {
			return err
		}

		if len(entries) > 0 {
			return nil
		}

		return os.Remove(path)
	}

	return nil
}
