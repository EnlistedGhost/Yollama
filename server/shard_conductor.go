package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
	"log/slog"

	"github.com/EnlistedGhost/Yollama/manifest"
	"github.com/EnlistedGhost/Yollama/types/model"
)

// HFFile represents an individual file entry returned by the Hugging Face tree API
type HFFile struct {
	Type string `json:"type"` 
	Path string `json:"path"` 
	Size int64  `json:"size"` 
}

// PullShardedGGUFManifest orchestrates downloading all pieces of a sharded model,
// merging them via the local toolchain, and returning a clean manifest back to PullModel.
func PullShardedGGUFManifest(ctx context.Context, n model.Name, regOpts *registryOptions) (*manifest.Manifest, []byte, error) {
	// 1. Discover remote paths
	shardPaths, err := fetchAndFilterHFShards(ctx, n)
	if err != nil {
		return nil, nil, err
	}

	// Create temporary staging
	baseBlobDir, err := manifest.BlobsPath("")
	if err != nil {
		return nil, nil, fmt.Errorf("failed to locate layout path for local content storage: %w", err)
	}
	
	stageDir := filepath.Join(baseBlobDir, "yollama-shard-stage")
	if err := os.MkdirAll(stageDir, 0o755); err != nil {
		return nil, nil, fmt.Errorf("failed to create staging directory on main storage pool: %w", err)
	}

	defer os.RemoveAll(stageDir) 
	slog.Info(fmt.Sprintf("[CONDUCTOR] Staging directory initialized at: %s", stageDir))

	// Download each shard file
	for i, shardPath := range shardPaths {
		slog.Info(fmt.Sprintf("[Shard %d/%d] Beginning download for: %s", i+1, len(shardPaths), path.Base(shardPath)))
		err := downloadShardFile(ctx, n, shardPath, stageDir, regOpts)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to download shard %s: %w", path.Base(shardPath), err)
		}
	}

	slog.Info("[CONDUCTOR] All shards downloaded successfully. Transferring control to toolchain pipeline...")

	// 4. Feed the staged folder straight into our direct-merge function
	var localFiles []string
	for _, shardPath := range shardPaths {
		localFiles = append(localFiles, filepath.Join(stageDir, path.Base(shardPath)))
	}

	assembledDigest, err := buildModelFromStagedFiles(ctx, localFiles)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to compile and merge shards: %w", err)
	}

	// 5. Generate a synthetic manifest structure to hand back to PullModel
	mf, manifestBytes, err := generateSyntheticManifest(assembledDigest)
	return mf, manifestBytes, err
}

// downloadShardFile fetches a single file from the HF repository and streams it to the staging directory
func downloadShardFile(ctx context.Context, n model.Name, shardPath string, destDir string, regOpts *registryOptions) error {
	// Directly target the asset download gateway endpoint
	modelURL := n.BaseURL()
	
	// Chain the direct asset download gateway verbs onto your verified path
	// This results in: https://{host}/{namespace}/{model}/resolve/main/{shardPath}
	downloadURL := modelURL.JoinPath(n.DisplayNamespaceModel(), "resolve", "main", shardPath)
	
	slog.Info(fmt.Sprintf("[CONDUCTOR] Pulling raw binary shard from: %s", downloadURL.String()))

	headers := make(http.Header)
	resp, err := makeRequestWithRetry(ctx, http.MethodGet, downloadURL, headers, nil, regOpts)
	if err != nil {
		return fmt.Errorf("network request failed for shard: %w", err)
	}
	defer resp.Body.Close()

	outFilePath := filepath.Join(destDir, path.Base(shardPath))
	outFile, err := os.Create(outFilePath)
	if err != nil {
		return fmt.Errorf("failed to create local staging file: %w", err)
	}
	defer outFile.Close()

	_, err = io.Copy(outFile, resp.Body)
	if err != nil {
		return fmt.Errorf("error occurred while writing shard data to disk: %w", err)
	}

	slog.Info(fmt.Sprintf("[CONDUCTOR] Successfully cached shard: %s", path.Base(shardPath)))
	return nil
}

// fetchAndFilterHFShards queries the Hugging Face file tree API and returns the matching shard paths
func fetchAndFilterHFShards(ctx context.Context, n model.Name) ([]string, error) {
	// 1. Leverage your verified base URL pattern
	modelURL := n.BaseURL()
	
	// 2. Chain the target endpoints natively
	requestURL := modelURL.JoinPath("api", "models", n.DisplayNamespaceModel(), "tree", "main")
	
	// 3. NEW: Inject the recursive query parameter to scan deep subdirectories!
	query := requestURL.Query()
	query.Set("recursive", "true")
	requestURL.RawQuery = query.Encode()
	
	slog.Info(fmt.Sprintf("[CONDUCTOR] Querying deep repository tree JSON: %s", requestURL.String()))
	
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL.String(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	
	// Use standard default client
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("hf api tree request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("hf api returned status %d: %s", resp.StatusCode, string(body))
	}

	var allFiles []HFFile
	if err := json.NewDecoder(resp.Body).Decode(&allFiles); err != nil {
		return nil, fmt.Errorf("failed to unmarshal hf repository layout: %w", err)
	}

	// Prepare clean, case-insensitive target signatures for the tag
	// This generates pure strings like "q2_0_l" and "q2-0-l" to compare against
	tagWithUnderscore := strings.ToLower(n.Tag)
	tagWithHyphen := strings.ReplaceAll(tagWithUnderscore, "_", "-")

	var matchingShards []string
	for _, file := range allFiles {
		// Ignore directories or metadata tracker files
		if file.Type != "file" {
			continue
		}

		lowerPath := strings.ToLower(file.Path)

		// Rule A: The path MUST end with the standard shard file suffix layout
		// Rule B: The path MUST contain either the underscore or hyphen version of your tag
		if strings.HasSuffix(lowerPath, ".gguf") && 
		   strings.Contains(lowerPath, "-of-") && 
		   (strings.Contains(lowerPath, tagWithUnderscore) || strings.Contains(lowerPath, tagWithHyphen)) {
			
			// Retain the pristine casing of the original path mapping for Hugging Face's API
			matchingShards = append(matchingShards, file.Path)
			slog.Info(fmt.Sprintf("[CONDUCTOR] -> Successfully Isolated Shard: %s", path.Base(file.Path)))
		}
	}

	if len(matchingShards) == 0 {
		return nil, fmt.Errorf("could not find split GGUF shards for tag '%s'", n.Tag)
	}

	slog.Info(fmt.Sprintf("[CONDUCTOR] Successfully mapped all %d fragments for processing.", len(matchingShards)))
	return matchingShards, nil
}

// buildModelFromStagedFiles calls your specific compiled llama-gguf-split tool
// to merge files directly to their destination without temporary copies.
func buildModelFromStagedFiles(ctx context.Context, filePaths []string) (string, error) {
	slog.Info("[CONDUCTOR] Activating direct-to-disk toolchain assembly layer...")

	var firstShard string
	for _, fp := range filePaths {
		if strings.Contains(fp, "-00001-of-") {
			firstShard = fp
			break
		}
	}

	if firstShard == "" {
		return "", fmt.Errorf("missing starting sequence split (-00001-of-...) in target directory")
	}

	baseBlobDir, err := manifest.BlobsPath("")
	if err != nil {
		return "", fmt.Errorf("failed to locate layout path for local content storage: %w", err)
	}
	if err := os.MkdirAll(baseBlobDir, 0o755); err != nil {
		return "", err
	}

	interimTargetFile := filepath.Join(baseBlobDir, "yollama_direct_assembly.gguf")

	binaryPath := "/usr/local/lib/yollama/llama-gguf-split"
	cmd := exec.CommandContext(ctx, binaryPath, "--merge", firstShard, interimTargetFile)

	var errLog strings.Builder
	cmd.Stderr = &errLog

	slog.Info(fmt.Sprintf("[CONDUCTOR] Forking process: %s --merge %s %s", 
		filepath.Base(binaryPath), filepath.Base(firstShard), interimTargetFile))
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("binary compilation failed: %s: %w", errLog.String(), err)
	}

	assembledFile, err := os.Open(interimTargetFile)
	if err != nil {
		return "", fmt.Errorf("failed to open assembled output model tracking reference: %w", err)
	}

	digest, _, err := func() (string, int64, error) {
		defer assembledFile.Close()
		h, size := GetSHA256Digest(assembledFile)
		return h, size, nil
	}()
	if err != nil {
		return "", err
	}

	finalDestinationPath, err := manifest.BlobsPath(digest)
	if err != nil {
		return "", fmt.Errorf("failed to project internal storage mapping signature: %w", err)
	}

	slog.Info(fmt.Sprintf("[CONDUCTOR] Fast-linking compiled architecture asset to signature: %s", digest))
	if err := os.Rename(interimTargetFile, finalDestinationPath); err != nil {
		os.Remove(interimTargetFile)
		return "", fmt.Errorf("failed to bind asset into formal content pool: %w", err)
	}

	return digest, nil
}

// generateSyntheticManifest generates a valid synthetic OCI model manifest payload matching what PullModel expects.
func generateSyntheticManifest(modelDigest string) (*manifest.Manifest, []byte, error) {
	blobPath, err := manifest.BlobsPath(modelDigest)
	if err != nil {
		return nil, nil, err
	}
	fileInfo, err := os.Stat(blobPath)
	if err != nil {
		return nil, nil, fmt.Errorf("compiled blob file tracking lost: %w", err)
	}

	// FIXED: Dropped unknown Created field to match structural limits
	configData := model.ConfigV2{
		Architecture: "llama", 
	}
	configBytes, err := json.Marshal(configData)
	if err != nil {
		return nil, nil, err
	}
	configDigest, _ := GetSHA256Digest(bytes.NewReader(configBytes))

	configBlobPath, err := manifest.BlobsPath(configDigest)
	if err == nil {
		_ = os.WriteFile(configBlobPath, configBytes, 0o644)
	}

	mf := &manifest.Manifest{
		Config: manifest.Layer{
			MediaType: "application/vnd.docker.container.image.v1+json",
			Digest:    configDigest,
			Size:      int64(len(configBytes)),
		},
		Layers: []manifest.Layer{
			{
				MediaType: "application/vnd.yollama.image.model",
				Digest:    modelDigest,
				Size:      fileInfo.Size(),
			},
		},
	}

	manifestBytes, err := json.Marshal(mf)
	if err != nil {
		return nil, nil, err
	}

	return mf, manifestBytes, nil
}
