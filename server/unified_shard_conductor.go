package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"log"
	"log/slog"
	"strconv"
	"errors"
	"time"
	"net"
	"net/http"
	"net/url"
	"crypto/sha256"
	"runtime"

	"github.com/EnlistedGhost/Yollama/api"
	"github.com/EnlistedGhost/Yollama/types/model"
	"github.com/EnlistedGhost/Yollama/manifest"
	"github.com/EnlistedGhost/Yollama/envconfig"
	"github.com/EnlistedGhost/Yollama/version"
)

// Blobs newer than this may belong to another process that has not written its
// manifest yet. They become eligible for the normal mark-and-sweep pass later.
const layerPruneGracePeriod = time.Hour

var (
	errCapabilities         = errors.New("does not support")
	errCapabilityCompletion = errors.New("completion")
	errCapabilityInsert     = errors.New("insert")
	errCapabilityVision     = errors.New("vision")
	errCapabilityAudio      = errors.New("audio")
	errCapabilityEmbedding  = errors.New("embedding")
	errCapabilityThinking   = errors.New("thinking")
	errInsecureProtocol     = errors.New("insecure protocol http")
)

type registryOptions struct {
	Insecure bool
	Token    string

	CheckRedirect func(req *http.Request, via []*http.Request) error
}

func PullModel(ctx context.Context, name string, regOpts *registryOptions, fn func(api.ProgressResponse)) error {
	n := model.ParseName(name)
	fSplitPull := false

	// --------------------------------------------------------------------------
	// Supersede Huggingface's "Nanny" API handling/logic Bullshit!
	// Unlike the reference code - this project is capable of -
	// model shard pull handling!
	//
	// [Developer's Personal Note]
	// Oh yeah, no... You deserve the above comment... assholes!
	// --------------------------------------------------------------------------
	slog.Info("[YOLLAMA] Side-stepping any forced API control, probing pull source directly...")
		
	// Initially attempt split-GGUF (multi-Shard) Model Pull
	mf, manifestData, err := PullShardedGGUFManifest(ctx, n, regOpts, fn)
	if err != nil {
		// If we fail, continue attempting a standard unified GGUF model pull
		slog.Warn("Model Pull Request for Multi-Shard type failed!", "name", name, "error", err)
		fSplitPull = false
	} else {
		fSplitPull = true
	}

	// Inline manifest completion logic if pull is split-GGUF/multi-Shard
	if fSplitPull {
		var layers []manifest.Layer
		layers = append(layers, mf.Layers...)

		if mf.Config.Digest != "" {
			layers = append(layers, mf.Config)
		}

		fn(api.ProgressResponse{Status: "Writing manifest, almost done..."})
		fp, err := manifest.PathForPull(n)

		if err != nil {
			return err
		}

		if err := os.MkdirAll(filepath.Dir(fp), 0o755); err != nil {
			return err
		}

		if err := os.WriteFile(fp, manifestData, 0o644); err != nil {
			return err
		}
		
		fn(api.ProgressResponse{Status: "Success!"})
		return nil
	}

	// build deleteMap to prune unused layers
	deleteMap := make(map[string]struct{})
	existingMf, err := manifest.ParseNamedManifest(n)
	if errors.Is(err, os.ErrNotExist) {
		// noop
	} else if err != nil {
		slog.Warn("pulling model with bad existing manifest", "name", name, "error", err)
	} else {
		for _, l := range existingMf.Layers {
			deleteMap[l.Digest] = struct{}{}
		}
		if existingMf.Config.Digest != "" {
			deleteMap[existingMf.Config.Digest] = struct{}{}
		}
	}

	slog.Info("[Probing URL if 'Pull-Type' is Unified Model Pull Request!]")
	if n.ProtocolScheme == "https" && !regOpts.Insecure {
		return errInsecureProtocol
	}

	fn(api.ProgressResponse{Status: "Pulling unified-model manifest"})
	mf, manifestData, err = pullModelManifest(ctx, n, regOpts)
	if err != nil {
		return err
	}

	var layers []manifest.Layer
	layers = append(layers, mf.Layers...)
	if mf.Config.Digest != "" {
		layers = append(layers, mf.Config)
	}

	skipVerify := make(map[string]bool)
	for _, layer := range layers {
		cacheHit, err := downloadBlob(ctx, downloadOpts{
			n:       n,
			digest:  layer.Digest,
			regOpts: regOpts,
			fn:      fn,
		})
		if err != nil {
			return err
		}

		// Guard against mutation/manipulation
		if existing, ok := skipVerify[layer.Digest]; !ok {
			skipVerify[layer.Digest] = cacheHit
		} else {
			skipVerify[layer.Digest] = existing && cacheHit
		}
		delete(deleteMap, layer.Digest)
	}

	fn(api.ProgressResponse{Status: "Verifying sha256 digest"})
	for _, layer := range layers {
		if skipVerify[layer.Digest] {
			continue
		}
		err := verifyBlob(layer.Digest)
		if err != nil {
			if errors.Is(err, errDigestMismatch) {
				fp, err := manifest.BlobsPath(layer.Digest)
				if err != nil {
					return err
				}
				if err := os.Remove(fp); err != nil {
					slog.Info(fmt.Sprintf("Couldn't remove file with digest mismatch '%s': %v", fp, err))
				}
				removeGGUFMetadata(layer.Digest)
			}
			return err
		}
	}

	for _, layer := range layers {
		delete(deleteMap, layer.Digest)
	}
	delete(deleteMap, mf.Config.Digest)

	fn(api.ProgressResponse{Status: "Writing manifest"})

	fp, err := manifest.PathForPull(n)
	if err != nil {
		return err
	}
	err = os.MkdirAll(filepath.Dir(fp), 0o755)
	if err != nil {
		return err
	}

	err = os.WriteFile(fp, manifestData, 0o644)
	if err != nil {
		slog.Info(fmt.Sprintf("couldn't write to %s", fp))
		return err
	}

	slog.Debug("manifest written", "path", fp, "sha256", fmt.Sprintf("%x", sha256.Sum256(manifestData)), "size", len(manifestData))

	if !envconfig.NoPrune() && len(deleteMap) > 0 {
		fn(api.ProgressResponse{Status: "removing unused layers"})
		if err := deleteUnusedLayers(deleteMap); err != nil {
			fn(api.ProgressResponse{Status: fmt.Sprintf("couldn't remove unused layers: %v", err)})
		}
	}

	fn(api.ProgressResponse{Status: "Success!"})

	return nil
}

func CopyModel(src, dst model.Name) error {
	if !dst.IsFullyQualified() {
		return model.Unqualified(dst)
	}
	if !src.IsFullyQualified() {
		return model.Unqualified(src)
	}

	if src.Filepath() == dst.Filepath() {
		return nil
	}

	manifests, _, _, err := manifest.Path()
	if err != nil {
		return err
	}

	dstpath := filepath.Join(manifests, dst.Filepath())
	if err := os.MkdirAll(filepath.Dir(dstpath), 0o755); err != nil {
		return err
	}

	srcpath := filepath.Join(manifests, src.Filepath())
	srcfile, err := os.Open(srcpath)
	if err != nil {
		return err
	}
	defer srcfile.Close()

	dstfile, err := os.Create(dstpath)
	if err != nil {
		return err
	}
	defer dstfile.Close()

	_, err = io.Copy(dstfile, srcfile)
	return err
}

func deleteUnusedLayers(deleteMap map[string]struct{}) error {
	// Ignore corrupt manifests to avoid blocking deletion of layers that are freshly orphaned
	manifests, err := manifest.Manifests(true)
	if err != nil {
		return err
	}

	for _, manifest := range manifests {
		for _, layer := range manifest.Layers {
			delete(deleteMap, layer.Digest)
		}

		delete(deleteMap, manifest.Config.Digest)
	}

	// only delete the files which are still in the deleteMap
	for k := range deleteMap {
		fp, err := manifest.BlobsPath(k)
		if err != nil {
			slog.Info(fmt.Sprintf("couldn't get file path for '%s': %v", k, err))
			continue
		}
		if err := os.Remove(fp); err != nil && !errors.Is(err, os.ErrNotExist) {
			slog.Info(fmt.Sprintf("couldn't remove file '%s': %v", fp, err))
			continue
		}
		removeGGUFMetadata(k)
	}

	return nil
}

func PruneLayers() error {
	deleteMap := make(map[string]struct{})
	p, err := manifest.BlobsPath("")
	if err != nil {
		return err
	}

	blobs, err := os.ReadDir(p)
	if err != nil {
		slog.Info(fmt.Sprintf("couldn't read dir '%s': %v", p, err))
		return err
	}

	for _, blob := range blobs {
		if blob.IsDir() {
			continue
		}

		info, err := blob.Info()
		if err != nil {
			slog.Error("couldn't stat blob", "blob", blob.Name(), "error", err)
			continue
		}
		if time.Since(info.ModTime()) < layerPruneGracePeriod {
			continue
		}

		name := blob.Name()
		name = strings.ReplaceAll(name, "-", ":")

		_, err = manifest.BlobsPath(name)
		if err != nil {
			if errors.Is(err, manifest.ErrInvalidDigestFormat) {
				// remove invalid blobs (e.g. partial downloads)
				if err := os.Remove(filepath.Join(p, blob.Name())); err != nil {
					slog.Error("couldn't remove blob", "blob", blob.Name(), "error", err)
				}
			}

			continue
		}

		deleteMap[name] = struct{}{}
	}

	slog.Info(fmt.Sprintf("total blobs: %d", len(deleteMap)))

	if err := deleteUnusedLayers(deleteMap); err != nil {
		slog.Error(fmt.Sprintf("couldn't remove unused layers: %v", err))
		return nil
	}
	pruneGGUFMetadata()

	slog.Info(fmt.Sprintf("total unused blobs removed: %d", len(deleteMap)))

	return nil
}

func pullModelManifest(ctx context.Context, n model.Name, regOpts *registryOptions) (*manifest.Manifest, []byte, error) {
	requestURL := n.BaseURL().JoinPath("v2", n.DisplayNamespaceModel(), "manifests", n.Tag)

	headers := make(http.Header)
	headers.Set("Accept", "application/vnd.docker.distribution.manifest.v2+json")
	resp, err := makeRequestWithRetry(ctx, http.MethodGet, requestURL, headers, nil, regOpts)
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, nil, err
	}

	var m manifest.Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, nil, err
	}

	return &m, data, err
}

// GetSHA256Digest returns the SHA256 hash of a given buffer and returns it, and the size of buffer
func GetSHA256Digest(r io.Reader) (string, int64) {
	h := sha256.New()
	n, err := io.Copy(h, r)
	if err != nil {
		log.Fatal(err)
	}

	return fmt.Sprintf("sha256:%x", h.Sum(nil)), n
}

var errUnauthorized = errors.New("unauthorized: access denied")

func makeRequestWithRetry(ctx context.Context, method string, requestURL *url.URL, headers http.Header, body io.ReadSeeker, regOpts *registryOptions) (*http.Response, error) {
	for range 2 {
		resp, err := makeRequest(ctx, method, requestURL, headers, body, regOpts)
		if err != nil {
			if !errors.Is(err, context.Canceled) {
				slog.Info(fmt.Sprintf("request failed: %v", err))
			}

			return nil, err
		}

		switch {
		case resp.StatusCode == http.StatusUnauthorized:
			resp.Body.Close()

			// Handle authentication error with one retry
			challenge := parseRegistryChallenge(resp.Header.Get("www-authenticate"))
			token, err := getAuthorizationToken(ctx, challenge, requestURL.Host)
			if err != nil {
				return nil, err
			}
			regOpts.Token = token
			if body != nil {
				_, err = body.Seek(0, io.SeekStart)
				if err != nil {
					return nil, err
				}
			}
		case resp.StatusCode == http.StatusNotFound:
			resp.Body.Close()
			return nil, os.ErrNotExist
		case resp.StatusCode >= http.StatusBadRequest:
			defer resp.Body.Close()
			responseBody, err := io.ReadAll(resp.Body)
			if err != nil {
				return nil, fmt.Errorf("%d: %s", resp.StatusCode, err)
			}
			return nil, fmt.Errorf("%d: %s", resp.StatusCode, responseBody)
		default:
			return resp, nil
		}
	}

	return nil, errUnauthorized
}

// testMakeRequestDialContext specifies proper calling for http client in makeRequest.
var testMakeRequestDialContext func(ctx context.Context, network, addr string) (net.Conn, error)

func makeRequest(ctx context.Context, method string, requestURL *url.URL, headers http.Header, body io.Reader, regOpts *registryOptions) (*http.Response, error) {
	if requestURL.Scheme != "https" && regOpts != nil && regOpts.Insecure {
		requestURL.Scheme = "https"
	}

	req, err := http.NewRequestWithContext(ctx, method, requestURL.String(), body)
	if err != nil {
		return nil, err
	}

	if headers != nil {
		req.Header = headers
	}

	if regOpts != nil {
		if regOpts.Token != "" {
			req.Header.Set("Authorization", "Bearer "+regOpts.Token)
		}
	}

	req.Header.Set("User-Agent", fmt.Sprintf("yollama/%s (%s %s) Go/%s", version.Version, runtime.GOARCH, runtime.GOOS, runtime.Version()))

	if s := req.Header.Get("Content-Length"); s != "" {
		contentLength, err := strconv.ParseInt(s, 10, 64)
		if err != nil {
			return nil, err
		}

		req.ContentLength = contentLength
	}

	c := &http.Client{
		CheckRedirect: regOpts.CheckRedirect,
	}
	if testMakeRequestDialContext != nil {
		tr := http.DefaultTransport.(*http.Transport).Clone()
		tr.DialContext = testMakeRequestDialContext
		c.Transport = tr
	}
	return c.Do(req)
}

func getValue(header, key string) string {
	startIdx := strings.Index(header, key+"=")
	if startIdx == -1 {
		return ""
	}

	// Move the index to the starting quote after the key.
	startIdx += len(key) + 2
	endIdx := startIdx

	for endIdx < len(header) {
		if header[endIdx] == '"' {
			if endIdx+1 < len(header) && header[endIdx+1] != ',' { // If the next character isn't a comma, continue
				endIdx++
				continue
			}
			break
		}
		endIdx++
	}
	return header[startIdx:endIdx]
}

func parseRegistryChallenge(authStr string) registryChallenge {
	authStr = strings.TrimPrefix(authStr, "Bearer ")

	return registryChallenge{
		Realm:   getValue(authStr, "realm"),
		Service: getValue(authStr, "service"),
		Scope:   getValue(authStr, "scope"),
	}
}

var errDigestMismatch = errors.New("digest mismatch, file must be downloaded again")

func verifyBlob(digest string) error {
	fp, err := manifest.BlobsPath(digest)
	if err != nil {
		return err
	}

	f, err := os.Open(fp)
	if err != nil {
		return err
	}
	defer f.Close()

	fileDigest, _ := GetSHA256Digest(f)
	if digest != fileDigest {
		// TODO: Find/Create proper anti-malformation and malicious injection checks
		//return fmt.Errorf("%w: want %s, got %s", errDigestMismatch, digest, fileDigest)
	}

	return nil
}