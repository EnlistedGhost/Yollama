// llama_server.go wraps the llama-server binary as a subprocess
//
// Yollama uses two chat paths with llama-server. Models with explicit Yollama
// renderers/parsers, Harmony handling, MLX, or an enabled Go TEMPLATE layer
// still render prompts in Go and call /completion. Other GGUF chat models use
// llama-server's chat_template handling through /v1/chat/completions.
//
// For structured output, JSON schemas are passed directly to llama-server via
// its json_schema field (avoiding the CGO SchemaToGrammar dependency). Raw BNF
// grammars are passed via the grammar field.
//
// llama-server auto-detects GPU layers (-ngl), thread count (-t), and flash
// attention (--flash-attn).
package llm

import (
	"bufio"
	"syscall"
	"bytes"
	"context"
	crand "crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"log/slog"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
)

// DefaultModelNumBatch is the default NumBatch used for embedding models
// when neither the model nor the request specifies num_batch.
const (
	DefaultModelNumBatch             = 1024
)

// DefaultModelNumBatchForContext caps the embedding batch default to the
// active context length before it is passed to llama-server.
func DefaultModelNumBatchForContext(numCtx int) int {
	if numCtx > 0 {
		return min(DefaultModelNumBatch, numCtx)
	}
	return DefaultModelNumBatch
}

// WithDefaultModelNumBatch applies the llama-server embedding batch
// default to a copy of opts.
func WithDefaultModelNumBatch(opts api.Options) api.Options {
	opts.NumBatch = DefaultModelNumBatchForContext(opts.NumCtx)
	return opts
}

// llamaServerRunner wraps an upstream llama-server process and implements the LlamaServer interface.
// It communicates with llama-server over HTTP.
type llamaServerRunner struct {
	port               int
	cmd                *exec.Cmd
	done               chan struct{}
	doneErr            error
	client             *http.Client
	memoryMu           sync.RWMutex
	memTotal           uint64 // actual total buffer size parsed from llama-server logs (bytes)
	memGPU             uint64 // actual GPU buffer size parsed from llama-server logs (bytes)
	memModelFileBacked uint64 // model weight bytes whose buffers mirror the on-disk file (mmap views + direct device copies); excludes repacked copies like CPU_REPACK
	memCPUMappedModel  uint64 // model weight bytes in mmap-backed CPU buffers (e.g. CPU_Mapped), parsed from llama-server logs
	gpuLayers          uint64 // model layers loaded on GPU, parsed from llama-server logs
	gpuLayerOverflow   int    // number of GPU-selected layers partially overflowed to CPU
	status             *StatusWriter
	options            api.Options
	modelPath          string
	mmprojMemory       uint64
	// mediaMarker must match the LLAMA_MEDIA_MARKER value passed to llama-server.
	// llama.cpp randomizes this by default; Yollama renders stable [img-N] markers
	// and rewrites them before forwarding the request.
	mediaMarker string

	// Per-device VRAM tracking, populated from llama-server log parsing.
	// Keys are device names from llama-server output (e.g., "CUDA0", "ROCm0", "MTL0").
	vramByDevice map[string]uint64

	// System-reported free VRAM per device at model load time, parsed from
	// "using device CUDA0 ... - 15221 MiB free" log lines. This reflects
	// real system state including external VRAM consumers (on platforms where
	// the GPU driver reports accurately). Keys match vramByDevice (e.g., "CUDA0").
	systemFreeAtLoad map[string]uint64

	// gpus is the list of GPU devices assigned to this runner at creation time,
	// used to map DeviceIDs to device names for VRAMByGPU lookups.
	gpus []ml.DeviceInfo

	ggml        *ggml.GGML
	totalLayers uint64 // maximum offloadable model layers
	loadStart   time.Time

	sem *semaphore.Weighted

	launch                  llamaServerLaunchConfig
	output                  *memoryParsingWriter
}

type llamaServerLaunchConfig struct {
	modelPath            string
	projectors           []string
	modelLayers          uint64
	mmprojMemory         uint64
	opts                 api.Options
	numParallel          int
	kvCacheType          string
	embedding            bool
	config               LlamaServerConfig
	gpus                 []ml.DeviceInfo
	gpuLibs              []string
	extraEnvs            map[string]string
}

func newLlamaServerHTTPClient() *http.Client {
	return &http.Client{
		Transport: &http.Transport{
			DisableKeepAlives: true,
			Proxy:             nil,
		},
	}
}

var defaultLlamaServerHTTPClient = newLlamaServerHTTPClient()

func (s *llamaServerRunner) httpClient() *http.Client {
	if s.client != nil {
		return s.client
	}
	return defaultLlamaServerHTTPClient
}

func (s *llamaServerRunner) ModelPath() string {
	return s.modelPath
}

func (s *llamaServerRunner) Pid() int {
	if s.cmd != nil && s.cmd.Process != nil {
		return s.cmd.Process.Pid
	}
	return 0
}

func (s *llamaServerRunner) GetPort() int {
	return s.port
}

func (s *llamaServerRunner) exitReason() string {
    if s.cmd == nil || s.cmd.ProcessState == nil {
        return ""
    }
    ws, ok := s.cmd.ProcessState.Sys().(syscall.WaitStatus)
    if !ok {
        return fmt.Sprintf("exit code %d", s.cmd.ProcessState.ExitCode())
    }
    if ws.Signaled() {
        return fmt.Sprintf("killed by signal %d", int(ws.Signal()))
    }
    return fmt.Sprintf("exited with code %d", s.cmd.ProcessState.ExitCode())
}

func (s *llamaServerRunner) HasExited() bool {
		reasonForExit := s.exitReason()
		if reasonForExit == "" {
			fmt.Sprintf("exit reason: UNKNOWN")
		}
    return s.cmd != nil && s.cmd.ProcessState != nil
}

func (s *llamaServerRunner) llamaServerMediaMarker() string {
	if s.mediaMarker != "" {
		return s.mediaMarker
	}
	return "<__media__>"
}

func newLlamaServerMediaMarker() string {
	var b [16]byte
	if _, err := crand.Read(b[:]); err == nil {
		return fmt.Sprintf("<__yollama_media_%x__>", b)
	}

	return fmt.Sprintf("<__yollama_media_%d_%d__>", time.Now().UnixNano(), rand.Int63())
}

func (s *llamaServerRunner) ContextLength() int {
	return s.options.NumCtx
}

func getLlamaPromptsPath() (string, error) {
    // 1. Get the dynamic home directory path
    homeUserDir, err := os.UserHomeDir()
    if err != nil {
        log.Fatalf("Error no such directory found: %v", err)
    }

    // 2. Safely join the home directory with the .yollama folder
    llamaPromptPath := filepath.Join(homeUserDir, ".yollama/prompts")
    fmt.Println("Yollama Prompts directory:", llamaPromptPath)

    return llamaPromptPath, err
}

func getLlamaMediaPath() (string, error) {
    // 1. Get the dynamic home directory path
    homeUserDir, err := os.UserHomeDir()
    if err != nil {
        log.Fatalf("Error no such directory found: %v", err)
    }

    // 2. Safely join the home directory with the .yollama folder
    llamaMediaPath := filepath.Join(homeUserDir, ".glassmorphic/.glassmorphism_media")
    fmt.Println("Yollama Media directory:", llamaMediaPath)

    return llamaMediaPath, err
}

// FindLlamaServer locates the llama-server binary in lib/yollama/.
// There is a single binary that dynamically loads GPU backends at runtime.
func FindLlamaServer() (string, error) {
	path, candidates, err := findLlamaCppBinary("llama-server", defaultLlamaCppBinarySearch())
	if err != nil {
		return "", fmt.Errorf("llama-server binary not found (checked: %s). Run 'cmake -S llama/server --preset cpu && cmake --build --preset cpu' first", strings.Join(candidates, ", "))
	}
	return path, nil
}

// startLlamaServer spawns the upstream llama-server process with appropriate CLI flags.
func startLlamaServer(launch llamaServerLaunchConfig, out io.Writer) (cmd *exec.Cmd, port int, err error) {
	cmd = nil

	exe, err := FindLlamaServer()
	if err != nil {
		return nil, 0, err
	}

	// Allocate a port
	port = 9229
	if a, err := net.ResolveTCPAddr("tcp", "localhost:0"); err == nil {
		var l *net.TCPListener
		if l, err = net.ListenTCP("tcp", a); err == nil {
			port = l.Addr().(*net.TCPAddr).Port
			l.Close()
		}
	}
	if port == 0 {
		slog.Debug("ResolveTCPAddr failed, using random port")
		port = 9229
	}

	pathToPrompts, err := getLlamaPromptsPath()
    if err != nil {
        slog.Error("[YOLLAMA] | ❌ Failed to resolve Saved-Prompts path.", "error", err)
        return cmd, port, nil
    } else {
        fmt.Printf("[YOLLAMA] | ✅ Fetched Saved-Prompts path: %s\n", pathToPrompts)
    }
    pathToMedia, err := getLlamaMediaPath()
    if err != nil {
        slog.Error("[YOLLAMA] | ❌ Failed to resolve Media path.", "error", err)
        return cmd, port, nil
    } else {
        fmt.Printf("[YOLLAMA] | ✅ Fetched Media path: %s\n", pathToMedia)
    }

	// NOTES: IGNORE THESE COMMENTS, they are for myself as notes only.
	//
	// Build CLI flags — minimal set, let llama-server auto-detect the rest
	// DO NOT cache prompt, that's bullshit to reuse my prompts (our prompts)
	// Fuck drafting
	params := []string{
		"--model", launch.modelPath,
		"--port", strconv.Itoa(port),
		"--host", "127.0.0.1",
		"--cache-ram", "0",
		"--no-webui",
		"--offline",
		"--no-warmup",
		"--no-repack",
		"--spec-draft-n-max", "0",
		"--ctx-checkpoints", "0",
		"--reasoning-budget", "-1",
		"--keep", "0",
		"--cache-reuse", "0",
		"--no-cache-prompt",
		"--kv-unified",
		"--no-context-shift",
		"--no-cache-idle-slots",
		"--sleep-idle-seconds", "-1",
		"--slot-prompt-similarity", "0.0",
		"--threads", "12",
		"--threads-http", "12",
		"--parallel", "2",
		"--split-mode", "layer",
		"-fa", "on",
		"--fit", "on",
		"--cache-type-k", "f16",
		"--cache-type-v", "f16",
		//"--agent",
		"--reasoning-format", "deepseek-legacy",
    	//"--video-ffmpeg-dir", "/home/sera/.glassmorphic/.glassmorphism_media",
	}

	// Append additional params
    params = append(params, "--log-prompts-dir", pathToPrompts)
    params = append(params, "--media-path", pathToMedia)
    params = append(params, "--load-mode", "mmap")
    //params = append(params, "--mmap")
    params = appendLlamaServerLogArgs(params)
	params = appendJinjaArgs(params, launch.config)
	params = appendMMProjArgs(params, launch)
	//params = appendFlashAttentionArgs(params, launch.gpus)
	params = appendCtxArgs(params, launch.opts)
	params = appendBatchArgs(params, launch.opts)
	params = appendMediaArgs(params, launch.opts)

	// Set up library paths for GPU backend discovery
	cmd = exec.Command(exe, params...)

	if out != nil {
		// os/exec serializes Write calls when stdout and stderr share a writer.
		cmd.Stdout = out
		cmd.Stderr = out
	}
	cmd.SysProcAttr = LlamaServerSysProcAttr
	SetupLlamaServerCommandEnv(cmd, exe, launch.gpuLibs, launch.extraEnvs)

	slog.Info("starting llama-server", "cmd", cmd)
	slog.Debug("subprocess", "", filteredEnv(cmd.Env))

	if err = cmd.Start(); err != nil {
		return nil, 0, err
	}
	return cmd, port, nil
}

// SetupLlamaServerCommandEnv configures the environment for a llama-server
// subprocess so discovery and real model runners use the same library search
// paths and GPU backend selection.
func SetupLlamaServerCommandEnv(cmd *exec.Cmd, exe string, gpuLibs []string, extraEnvs map[string]string) {
	cmd.Env = os.Environ()

	envUpdates := make(map[string]string, len(extraEnvs)+2)
	for k, v := range extraEnvs {
		envUpdates[k] = v
	}

	libraryPaths := llamaServerLibraryPaths(exe, gpuLibs, envUpdates)
	pathEnv := llamaServerLibraryPathEnv()
	envUpdates[pathEnv] = strings.Join(libraryPaths, string(filepath.ListSeparator))

	applied := make(map[string]bool, len(envUpdates))
	for i := range cmd.Env {
		key, _, ok := strings.Cut(cmd.Env[i], "=")
		if !ok {
			continue
		}
		for updateKey, updateVal := range envUpdates {
			if strings.EqualFold(key, updateKey) {
				cmd.Env[i] = updateKey + "=" + updateVal
				applied[updateKey] = true
			}
		}
	}
	for key, val := range envUpdates {
		if !applied[key] {
			cmd.Env = append(cmd.Env, key+"="+val)
		}
	}
}

func llamaServerLibraryPathEnv() string {
	switch runtime.GOOS {
	case "windows":
		return "PATH"
	case "darwin":
		return "DYLD_LIBRARY_PATH"
	default:
		return "LD_LIBRARY_PATH"
	}
}

func llamaServerLibraryPaths(exe string, gpuLibs []string, envUpdates map[string]string) []string {
	llamaDir := filepath.Dir(exe)
	seen := map[string]bool{}
	var libraryPaths []string
	addPath := func(path string) {
		if path == "" || seen[path] {
			return
		}
		seen[path] = true
		libraryPaths = append(libraryPaths, path)
	}

	// Library path ordering:
	// 1. llama-server's own directory — ggml-base, ggml-cpu, libllama
	// 2. GPU variant directories — cublas, cudart, backend DLL/.so
	// 3. User/system library path
	addPath(llamaDir)
	for _, dir := range gpuLibs {
		if dir == ml.LibYollamaPath || dir == llamaDir {
			continue
		}
		if envUpdates["GGML_BACKEND_PATH"] == "" {
			if backend := findLlamaServerGPUBackend(dir); backend != "" {
				envUpdates["GGML_BACKEND_PATH"] = backend
			}
		}
		addPath(dir)
	}
	if libraryPath, ok := os.LookupEnv(llamaServerLibraryPathEnv()); ok {
		for _, dir := range filepath.SplitList(libraryPath) {
			addPath(dir)
		}
	}
	return adjustPlatformLibraryPaths(libraryPaths, gpuLibs)
}

func findLlamaServerGPUBackend(dir string) string {
	patterns := []string{
		"libggml-*.so*",
		"libggml-*.dylib",
		"libggml-*.dll",
		"ggml-*.dll",
	}
	var candidates []string
	for _, pattern := range patterns {
		matches, _ := filepath.Glob(filepath.Join(dir, pattern))
		candidates = append(candidates, matches...)
	}
	slices.Sort(candidates)

	for _, match := range candidates {
		if isLlamaServerGPUBackend(match) {
			return match
		}
	}
	return ""
}

func isLlamaServerGPUBackend(path string) bool {
	name := strings.ToLower(filepath.Base(path))
	for _, prefix := range []string{
		"libggml-base",
		"ggml-base",
		"libggml-cpu",
		"ggml-cpu",
	} {
		if strings.HasPrefix(name, prefix) {
			return false
		}
	}
	return true
}

func appendLlamaServerLogArgs(params []string) []string {
	// Keep startup memory/offload lines visible for scheduler accounting.
	return append(params,
		"--log-verbosity", "4",
		"--no-log-prefix",
		"--no-log-timestamps",
		"--log-colors", "on",
	)
}

func appendMediaArgs(params []string, opts api.Options) []string {
	//--image, --audio, --video FILE
	// Hey... Hey go-lang? GO FUCK YOURSELF!
	return params
}

// getYollamaTLSConfigPath returns the full path to the yollama_tls.conf file.
func getYollamaCtxConfigPath() (string, error) {
    // 1. Get the dynamic home directory path
    homeUserDir, err := os.UserHomeDir()
    if err != nil {
        log.Fatalf("[YOLLAMA] | Ctx config Error: no such directory found: %v", err)
    }

    // 2. Safely join the home directory with the .yollama folder
    ollamaPath := filepath.Join(homeUserDir, ".yollama")
    fmt.Println("Yollama directory:", ollamaPath)

    ctxConfigPath := filepath.Join(ollamaPath, "yollama_ctx.conf")
    return ctxConfigPath, err
}

// readYollamaTLSConfig reads the config file and returns true if TLS is enabled.
// The file should contain either 1 (true) or 0 (false).
func readYollamaCtxConfig(ctxConfigPath string) (int, error) {
    // Read entire file into byte slice
    ctxConfigData, err := os.ReadFile(ctxConfigPath)
    if err != nil {
        return 0, fmt.Errorf("[YOLLAMA] | failed to read ctx file: %w", err)
    }

    // Convert bytes to string (trim whitespace and newlines)
    ctxValue := strings.TrimSpace(string(ctxConfigData))

    // Convert string to integer
    numCtx_config, err := strconv.Atoi(ctxValue)
    if err != nil {
        return 0, fmt.Errorf("[YOLLAMA] | error converting configured Ctx value str to num")
    }

    return numCtx_config, nil
}

func appendCtxArgs(params []string, opts api.Options) []string {
	pathCtxConfig, err := getYollamaCtxConfigPath()

    if err != nil {
        slog.Error("[YOLLAMA] | ❌ Failed to resolve Ctx config path.", "error", err)
		return params
    } else {
        fmt.Printf("[YOLLAMA] | ✅ Fetched configured Ctx config path: %s\n", pathCtxConfig)
    }

	// Read the Ctx value
    setLlamaCtx, err := readYollamaCtxConfig(pathCtxConfig)
    if err != nil {
        // Fail open to base default Ctx (we set 0 here for now)
        slog.Warn("[YOLLAMA] | ⚠️ Ctx config unreadable; defaulting to static numCtx.", "error", err)
    } else {
      	fmt.Printf("[YOLLAMA] | ✅ Fetched configured Ctx enabled value: %t\n", setLlamaCtx)
    }


	if setLlamaCtx > 0 {
		params = append(params, "-c", strconv.Itoa(setLlamaCtx))
	} else {
		setLlamaCtx = 29000 
	params = append(params, "-c", strconv.Itoa(setLlamaCtx))
	}

	return params
}

func appendBatchArgs(params []string, opts api.Options) []string {
	if opts.NumBatch > 0 {
		params = append(params, "-b", strconv.Itoa(opts.NumBatch), "-ub", strconv.Itoa(opts.NumBatch))
	} else {
		params = append(params, "-b", strconv.Itoa(1024), "-ub", strconv.Itoa(1024))
	}
	return params
}

func appendFlashAttentionArgs(params []string, gpus []ml.DeviceInfo) []string {
	return append(params, "--flash-attn", "auto")
}

func appendMMProjArgs(params []string, launch llamaServerLaunchConfig) []string {
	if len(launch.projectors) == 0 {
		return params
	}

	params = append(params, "--mmproj", launch.projectors[0])
	return params
}

func appendJinjaArgs(params []string, config LlamaServerConfig) []string {
	if config.DisableJinja {
		return append(params, "--no-jinja", "--chat-template", "chatml")
	}

	return params
}

func mmprojMemoryRequirement(modelPath string, f *ggml.GGML, projectors []string) (uint64, error) {
	if len(projectors) == 0 {
		return 0, nil
	}

	if projectors[0] == modelPath {
		if f == nil {
			return 0, errors.New("read inline mmproj metadata: missing model metadata")
		}
		var size uint64
		for _, prefix := range []string{"v.", "mm.", "a."} {
			for _, tensor := range f.Tensors().Items(prefix) {
				size += tensor.Size()
			}
		}
		if size == 0 {
			return 0, errors.New("read inline mmproj metadata: no projector tensors found")
		}
		return size, nil
	}

	file, err := os.Open(projectors[0])
	if err != nil {
		return 0, fmt.Errorf("read mmproj metadata %q: %w", projectors[0], err)
	}
	defer file.Close()

	projector, err := ggml.Decode(file, 1024)
	if err != nil {
		return 0, fmt.Errorf("read mmproj metadata %q: %w", projectors[0], err)
	}
	var size uint64
	for _, tensor := range projector.Tensors().Items() {
		size += tensor.Size()
	}
	if size == 0 {
		return 0, fmt.Errorf("read mmproj metadata %q: no projector tensors found", projectors[0])
	}
	return size, nil
}

// NewLlamaServerRunner creates a new llama-server runner that wraps the upstream llama-server binary.
func NewLlamaServerRunner(gpus []ml.DeviceInfo, modelPath string, f *ggml.GGML,projectors []string, opts api.Options, numParallel int, kvCacheType string, config LlamaServerConfig) (LlamaServer, error) {

	// Check if this is an embedding model
	arch := f.KV().Architecture()
	_, isEmbedding := f.KV()[fmt.Sprintf("%s.pooling_type", arch)]

	if len(projectors) == 0 && len(f.Tensors().Items("v.")) > 0 {
		projectors = []string{modelPath}
	}

	mmprojMemory, err := mmprojMemoryRequirement(modelPath, f, projectors)
	if err != nil {
		return nil, err
	}

	gpuLibs := ml.LibraryPaths(gpus)
	status := NewStatusWriter(os.Stderr)

	// memWriter wraps the status writer and parses buffer size lines from llama-server logs
	memWriter := &memoryParsingWriter{inner: status}

	mediaMarker := newLlamaServerMediaMarker()
	extraEnvs := ml.GetDevicesEnv(gpus)
	serverEnvs := make(map[string]string, len(extraEnvs)+1)
	for k, v := range extraEnvs {
		serverEnvs[k] = v
	}
	serverEnvs["LLAMA_MEDIA_MARKER"] = mediaMarker

	launch := llamaServerLaunchConfig{
		modelPath:    modelPath,
		projectors:   slices.Clone(projectors),
		modelLayers:  f.KV().BlockCount() + 1,
		mmprojMemory: mmprojMemory,
		opts:         opts,
		numParallel:  numParallel,
		kvCacheType:  kvCacheType,
		embedding:    isEmbedding,
		config:       config,
		gpus:         slices.Clone(gpus),
		gpuLibs:      slices.Clone(gpuLibs),
		extraEnvs:    cloneStringMap(serverEnvs),
	}

	s := &llamaServerRunner{
		client:           newLlamaServerHTTPClient(),
		status:           status,
		options:          opts,
		modelPath:        modelPath,
		mediaMarker:      mediaMarker,
		vramByDevice:     make(map[string]uint64),
		systemFreeAtLoad: make(map[string]uint64),
		gpus:             gpus,
		ggml:             f,
		totalLayers:      f.KV().BlockCount() + 1,
		sem:              semaphore.NewWeighted(int64(numParallel)),
		launch:           launch,
		output:           memWriter,
	}
	// Point the memory parsing writer at this runner so values are updated as logs stream in
	memWriter.runner = s

	if err := s.startProcess(); err != nil {
		msg := s.lastErrMsg()
		return nil, fmt.Errorf("error starting llama-server: %v %s", err, msg)
	}

	return s, nil
}

func cloneStringMap(src map[string]string) map[string]string {
	dst := make(map[string]string, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func (s *llamaServerRunner) startProcess() error {
	cmd, port, err := startLlamaServer(s.launch, s.output)
	if err != nil {
		return err
	}

	s.cmd = cmd
	s.port = port
	s.done = make(chan struct{})
	s.doneErr = nil
	s.loadStart = time.Now()

	// Reap subprocess when it exits.
	go func(cmd *exec.Cmd, done chan struct{}) {
		err := cmd.Wait()
		s.doneErr = err
		if msg := s.lastErrMsg(); err != nil && msg != "" {
			slog.Error("llama-server terminated", "error", err, "exit", ExitStatusFromError(err))
			s.doneErr = errors.New(msg)
		}
		close(done)
	}(s.cmd, s.done)

	return nil
}

// Load waits for llama-server to finish loading the model.
func (s *llamaServerRunner) Load(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, _ bool) ([]ml.DeviceID, error) {
	slog.Info("loading model via llama-server", "model", s.modelPath)

	if err := s.WaitUntilRunning(ctx); err != nil {
			return nil, fmt.Errorf("llama-server startup failed: %w", err)
	}

	// Verify that buffer size parsing captured GPU allocations.
	// If parsing failed (e.g., llama-server log format changed), warn so the
	// issue is visible in logs when users report problems.
	if len(s.gpus) > 0 && !s.hasParsedVRAM() {
		slog.Warn("llama-server VRAM tracking: no per-device buffer sizes were parsed from "+
			"llama-server logs. VRAM accounting will be inaccurate. This may indicate a "+
			"change in llama-server's log format — check for 'buffer size' lines in the output.",
			"model", s.modelPath, "gpus", len(s.gpus))
	}

	// Return device IDs for all GPUs when llama-server manages layer placement itself.
	deviceIDs := make([]ml.DeviceID, len(gpus))
	for i, g := range gpus {
		deviceIDs[i] = g.DeviceID
	}

	return deviceIDs, nil
}

func (s *llamaServerRunner) resetLoadAccounting() {
	s.memoryMu.Lock()
	defer s.memoryMu.Unlock()

	s.memTotal = 0
	s.memGPU = 0
	s.memModelFileBacked = 0
	s.memCPUMappedModel = 0
	s.gpuLayers = 0
	s.gpuLayerOverflow = 0
	for k := range s.vramByDevice {
		delete(s.vramByDevice, k)
	}
	for k := range s.systemFreeAtLoad {
		delete(s.systemFreeAtLoad, k)
	}
	if s.status != nil {
		s.status.SetLastError("")
	}
}

func (s *llamaServerRunner) hasParsedVRAM() bool {
	s.memoryMu.RLock()
	defer s.memoryMu.RUnlock()

	return len(s.vramByDevice) > 0
}

func (s *llamaServerRunner) getServerStatus(ctx context.Context) (ServerStatus, error) {
	if s.cmd.ProcessState != nil {
		msg := s.lastErrMsg()
		if s.cmd.ProcessState.ExitCode() == -1 {
			slog.Warn("llama-server process no longer running", "sys", s.cmd.ProcessState.Sys(), "string", s.cmd.ProcessState)
		}
		return ServerStatusError, fmt.Errorf("llama-server process no longer running: %s %s", ExitStatus(s.cmd.ProcessState.ExitCode()), msg)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("http://127.0.0.1:%d/health", s.port), nil)
	if err != nil {
		return ServerStatusError, fmt.Errorf("error creating health request: %v", err)
	}

	resp, err := s.httpClient().Do(req)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return ServerStatusNotResponding, errors.New("server not responding")
		}
		if strings.Contains(err.Error(), "connection refused") {
			return ServerStatusNotResponding, errors.New("connection refused")
		}
		return ServerStatusError, fmt.Errorf("health resp: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return ServerStatusError, fmt.Errorf("read health response: %w", err)
	}

	var result struct {
		Status string `json:"status"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return ServerStatusError, fmt.Errorf("health unmarshal: %w", err)
	}

	switch result.Status {
	case "ok":
		return ServerStatusReady, nil
	case "loading model":
		return ServerStatusLoadingModel, nil
	case "no slot available":
		return ServerStatusNoSlotsAvailable, nil
	default:
		return ServerStatusError, fmt.Errorf("llama-server error: %s", string(body))
	}
}

func (s *llamaServerRunner) getServerStatusRetry(ctx context.Context) (ServerStatus, error) {
	var retries int
	for {
		status, err := s.getServerStatus(ctx)
		if err != nil {
			return status, err
		}
		if status == ServerStatusNoSlotsAvailable {
			if retries >= 10 {
				return status, fmt.Errorf("no slots available after %d retries", retries)
			}
			time.Sleep(5 * time.Millisecond)
			retries++
			continue
		}
		return status, nil
	}
}

func (s *llamaServerRunner) Ping(ctx context.Context) error {
	_, err := s.getServerStatus(ctx)
	if err != nil {
		slog.Debug("llama-server unhealthy", "error", err)
	}
	return err
}

func (s *llamaServerRunner) WaitUntilRunning(ctx context.Context) error {
	loadDeadline := time.Now().Add(envconfig.LoadTimeout())

	slog.Info("waiting for llama-server to start responding")
	var lastStatus ServerStatus = -1

	for {
		select {
		case <-ctx.Done():
			slog.Warn("client connection closed before llama-server finished loading, aborting load")
			return fmt.Errorf("timed out waiting for llama-server to start: %w", ctx.Err())
		case <-s.done:
			if msg := s.lastErrMsg(); msg != "" {
				if s.doneErr == nil {
					return fmt.Errorf("llama-server process has terminated: %s", msg)
				}
				if s.cmd != nil && s.cmd.ProcessState != nil && s.cmd.ProcessState.ExitCode() >= 0 {
					return fmt.Errorf("llama-server process has terminated: %s: %s", ExitStatus(s.cmd.ProcessState.ExitCode()), msg)
				}
				if exit := ExitStatusFromError(s.doneErr); exit.Known() {
					return fmt.Errorf("llama-server process has terminated: %s: %s", exit, msg)
				}
				return fmt.Errorf("llama-server process has terminated: %w: %s", s.doneErr, msg)
			}
			if s.doneErr == nil {
				if s.cmd != nil && s.cmd.ProcessState != nil {
					return fmt.Errorf("llama-server process has terminated: %s", ExitStatus(s.cmd.ProcessState.ExitCode()))
				}
				return errors.New("llama-server process has terminated")
			}
			if exit := ExitStatusFromError(s.doneErr); exit.Known() {
				return fmt.Errorf("llama-server process has terminated: %s", exit)
			}
			return fmt.Errorf("llama-server process has terminated: %w", s.doneErr)
		default:
		}

		if time.Now().After(loadDeadline) {
			msg := s.lastErrMsg()
			return fmt.Errorf("timed out waiting for llama-server to start - %s", msg)
		}

		if s.cmd.ProcessState != nil {
			msg := s.lastErrMsg()
			return fmt.Errorf("llama-server process no longer running: %s %s", ExitStatus(s.cmd.ProcessState.ExitCode()), msg)
		}

		pollCtx, cancel := context.WithTimeout(ctx, 200*time.Millisecond)
		status, statusErr := s.getServerStatus(pollCtx)
		cancel()

		statusChanged := lastStatus != status
		if statusChanged && status != ServerStatusReady {
			slog.Info("waiting for llama-server to become available", "status", status)
		}

		switch status {
		case ServerStatusReady:
			if s.status != nil {
				s.status.SetLastError("")
			}
			slog.Info(fmt.Sprintf("llama-server started in %0.2f seconds", time.Since(s.loadStart).Seconds()))
			return nil
		case ServerStatusError:
			msg := s.lastErrMsg()
			if isRecoverableOutOfMemoryMessage(msg) || isRecoverableOutOfMemory(statusErr) {
				lastStatus = status
				time.Sleep(time.Millisecond * 250)
				continue
			}
			if IsOutOfMemoryMessage(msg) {
				return fmt.Errorf("llama-server reported out-of-memory during startup: %s", msg)
			}
			if IsOutOfMemory(statusErr) {
				return fmt.Errorf("llama-server reported out-of-memory during startup: %w", statusErr)
			}
			lastStatus = status
			time.Sleep(time.Millisecond * 250)
		default:
			lastStatus = status
			time.Sleep(time.Millisecond * 250)
		}
	}
}

func (s *llamaServerRunner) lastErrMsg() string {
	if s.status == nil {
		return ""
	}
	return s.status.LastError()
}

// llamaServerCompletionRequest is the request format for llama-server's POST /completion endpoint.
type llamaServerCompletionRequest struct {
	Prompt          any             `json:"prompt"`
	Stream          bool            `json:"stream"`
	CachePrompt     bool            `json:"cache_prompt"`
	NPredict        int             `json:"n_predict,omitempty"`
	NKeep           int             `json:"n_keep,omitempty"`
	Temperature     float32         `json:"temperature"`
	TopK            float32         `json:"top_k"`
	TopP            float32         `json:"top_p"`
	MinP            float32         `json:"min_p"`
	Stop            []string        `json:"stop,omitempty"`
	RepeatPenalty   float32         `json:"repeat_penalty"`
	RepeatLastN     int             `json:"repeat_last_n"`
	FreqPenalty     float32         `json:"frequency_penalty"`
	PresPenalty     float32         `json:"presence_penalty"`
	TypicalP        float32         `json:"typical_p,omitempty"`
	Seed            int             `json:"seed"`
	JsonSchema      json.RawMessage `json:"json_schema,omitempty"`
	NProbs          int             `json:"n_probs,omitempty"`
}

// llamaServerMultimodalPrompt is used when images are present.
// llama-server's /completion endpoint accepts this as the "prompt" field.
type llamaServerMultimodalPrompt struct {
	PromptString   string   `json:"prompt_string"`
	MultimodalData []string `json:"multimodal_data"`
}

// llamaServerCompletionResponse is the response format from llama-server's /completion endpoint.
type llamaServerCompletionResponse struct {
	Content                 string                 `json:"content"`
	Stop                    bool                   `json:"stop"`
	StopType                string                 `json:"stop_type"`
	Timings                 llamaServerTimings     `json:"timings"`
	CompletionProbabilities []llamaServerTokenProb `json:"completion_probabilities"`
}

type llamaServerChatChoice struct {
	Delta struct {
		Content          string `json:"content"`
		ReasoningContent string `json:"reasoning_content"`
	} `json:"delta"`
	FinishReason *string `json:"finish_reason"`
	Logprobs     struct {
		Content []llamaServerTokenProb `json:"content"`
	} `json:"logprobs"`
}

type llamaServerChatResponse struct {
	Choices []llamaServerChatChoice `json:"choices"`
	Timings llamaServerTimings      `json:"timings"`
	Error   any                     `json:"error"`
}

type llamaServerTimings struct {
	CacheN    int     `json:"cache_n"`
	PromptN   int     `json:"prompt_n"`
	PromptMS  float64 `json:"prompt_ms"`
	PredictN  int     `json:"predicted_n"`
	PredictMS float64 `json:"predicted_ms"`
}

func (t llamaServerTimings) promptEvalCount() int {
	return t.CacheN + t.PromptN
}

type llamaServerApplyTemplateResponse struct {
	Prompt string `json:"prompt"`
	Error  any    `json:"error"`
}

type llamaServerTokenProb struct {
    Token       string                 `json:"token"`
    Logprob     *float64               `json:"logprob,omitempty"`
    Prob        *float64               `json:"prob,omitempty"`
    TopLogprobs []llamaServerTokenProb `json:"top_logprobs,omitempty"`
    TopProbs    []llamaServerTokenProb `json:"top_probs,omitempty"`
}

func (s *llamaServerRunner) Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error {
	slog.Debug("llama-server completion request", "media", len(req.Media), "prompt_len", len(req.Prompt))

	if req.Options == nil {
		opts := api.DefaultOptions()
		req.Options = &opts
	}

	if err := s.sem.Acquire(ctx, 1); err != nil {
		if errors.Is(err, context.Canceled) {
			slog.Info("aborting completion request due to client closing the connection")
		}
		return err
	}
	defer s.sem.Release(1)

	req.Options.NumPredict = -1

	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return err
	} else if status != ServerStatusReady {
		return fmt.Errorf("unexpected server status: %s", status)
	}

	prompt := req.Prompt

	// Build the llama-server request
	lsReq := llamaServerCompletionRequest{
		Prompt:          prompt,
		Stream:          true,
		CachePrompt:     false,
		NPredict:        req.Options.NumPredict,
		NKeep:           req.Options.NumKeep,
		Temperature:     req.Options.Temperature,
		TopK:            req.Options.TopK,
		TopP:            req.Options.TopP,
		MinP:            req.Options.MinP,
		Stop:            req.Options.Stop,
		RepeatPenalty:   req.Options.RepeatPenalty,
		RepeatLastN:     req.Options.RepeatLastN,
		FreqPenalty:     req.Options.FrequencyPenalty,
		PresPenalty:     req.Options.PresencePenalty,
		TypicalP:        req.Options.TypicalP,
		Seed:            req.Options.Seed,
	}

	if req.Logprobs {
		lsReq.NProbs = max(req.TopLogprobs, 1)
	}

	// Convert media: replace Yollama's stable [img-N] markers with the per-process
	// llama-server media marker and package the matching payloads as base64.
	if len(req.Media) > 0 {
		promptStr := lsReq.Prompt.(string)
		var mediaData []string
		for _, media := range req.Media {
			marker := fmt.Sprintf("[img-%d]", media.ID)
			promptStr = strings.Replace(promptStr, marker, s.llamaServerMediaMarker(), 1)
			mediaData = append(mediaData, base64.StdEncoding.EncodeToString(media.Data))
		}
		lsReq.Prompt = llamaServerMultimodalPrompt{
			PromptString:   promptStr,
			MultimodalData: mediaData,
		}
	}

	buffer := &bytes.Buffer{}
	enc := json.NewEncoder(buffer)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(lsReq); err != nil {
		return fmt.Errorf("failed to marshal completion request: %v", err)
	}

	endpoint := fmt.Sprintf("http://127.0.0.1:%d/completion", s.port)
	serverReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, buffer)
	if err != nil {
		return fmt.Errorf("error creating completion request: %v", err)
	}
	serverReq.Header.Set("Content-Type", "application/json")

	res, err := s.httpClient().Do(serverReq)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return err
		}
		slog.Error("llama-server completion error", "error", err)
		if msg := s.lastErrMsg(); msg != "" {
			return fmt.Errorf("model runner has unexpectedly stopped, this may be due to resource limitations or an internal error, check yollama server logs for details: %s", msg)
		}
		return errors.New("model runner has unexpectedly stopped, this may be due to resource limitations or an internal error, check yollama server logs for details")
	}
	defer res.Body.Close()

	if res.StatusCode >= 400 {
		bodyBytes, err := io.ReadAll(res.Body)
		if err != nil {
			return fmt.Errorf("failed reading llama-server error response: %w", err)
		}

		return api.StatusError{StatusCode: res.StatusCode, ErrorMessage: s.statusErrorMessage(bodyBytes)}
	}

	// Parse SSE stream from llama-server. Delay the final Done callback until
	// after the response body is closed because routes may tokenize from that
	// callback to build the final Generate context.
	scanner := bufio.NewScanner(res.Body)
	buf := make([]byte, 0, llamaServerStreamInitialBufferSize)
	scanner.Buffer(buf, llamaServerStreamMaxBufferSize)

	var finalResp CompletionResponse
	var hasFinalResp bool

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}
			if bytes.HasPrefix(line, []byte(":")) {
				continue
			}

			evt, ok := bytes.CutPrefix(line, []byte("data: "))
			if !ok {
				evt = line
			}

			var lsResp llamaServerCompletionResponse
			if err := json.Unmarshal(evt, &lsResp); err != nil {
				return fmt.Errorf("error unmarshalling llama-server response: %v", err)
			}

			if lsResp.Content != "" && !lsResp.Stop {
				resp := CompletionResponse{
					Content: lsResp.Content,
				}
				resp.Logprobs = convertLogprobs(lsResp.CompletionProbabilities, req.TopLogprobs > 0)
				fn(resp)
			}

			if lsResp.Stop {
				doneReason := DoneReasonStop
				if lsResp.StopType == "limit" {
					doneReason = DoneReasonLength
				}

				finalResp = CompletionResponse{
					Content:            lsResp.Content,
					Done:               true,
					DoneReason:         doneReason,
					PromptEvalCount:    lsResp.Timings.promptEvalCount(),
					PromptEvalDuration: time.Duration(lsResp.Timings.PromptMS * float64(time.Millisecond)),
					EvalCount:          lsResp.Timings.PredictN,
					EvalDuration:       time.Duration(lsResp.Timings.PredictMS * float64(time.Millisecond)),
				}
				hasFinalResp = true
			}
		}

		if hasFinalResp {
			break
		}
	}

	if hasFinalResp {
		for scanner.Scan() {
		}

		if err := scanner.Err(); err != nil {
			if err := llamaServerStreamLimitError("response", err); err != nil {
				return err
			}
			return fmt.Errorf("error reading llama-server response: %v", err)
		}

		if err := res.Body.Close(); err != nil {
			return fmt.Errorf("error closing llama-server response: %v", err)
		}

		fn(finalResp)
		return nil
	}

	if err := scanner.Err(); err != nil {
		if err := llamaServerStreamLimitError("response", err); err != nil {
			return err
		}
		if strings.Contains(err.Error(), "unexpected EOF") || strings.Contains(err.Error(), "forcibly closed") {
			s.Close()
			msg := s.lastErrMsg()
			if msg == "" {
				msg = err.Error()
			}
			return fmt.Errorf("an error was encountered while running the model: %s", msg)
		}
		return fmt.Errorf("error reading llama-server response: %v", err)
	}

	return nil
}

func llamaServerStreamLimitError(label string, err error) error {
	if !strings.Contains(err.Error(), "token too long") {
		return nil
	}

	return fmt.Errorf("llama-server %s stream event exceeded %d MB limit", label, llamaServerStreamMaxBufferSize/(1000*1000))
}

func (s *llamaServerRunner) statusErrorMessage(body []byte) string {
	errMsg := strings.TrimSpace(string(body))
	statusMsg := s.lastErrMsg()
	if statusMsg == "" {
		return errMsg
	}

	if IsOutOfMemoryMessage(statusMsg) && !strings.Contains(strings.ToLower(errMsg), strings.ToLower(statusMsg)) {
		return strings.TrimSpace(errMsg + "\n" + statusMsg)
	}

	return errMsg
}

func convertLogprobs(probs []llamaServerTokenProb, includeTop bool) []Logprob {
    if len(probs) == 0 {
        return nil
    }
    result := make([]Logprob, len(probs))
    for i, p := range probs {
        var logprob float64
        switch {
        case p.Logprob != nil: // present → use it (even if the value is 0)
            logprob = *p.Logprob
        case p.Prob != nil:    // only prob was sent
            logprob = *p.Prob
        default:               // neither key arrived — don't guess a zero
            continue
        }

        result[i] = Logprob{
            TokenLogprob: TokenLogprob{Token: p.Token, Logprob: logprob},
        }

        if !includeTop {
            continue
        }

        var topProbs []llamaServerTokenProb
        switch {
        case p.TopLogprobs != nil:
            topProbs = p.TopLogprobs
        case p.TopProbs != nil:
            topProbs = p.TopProbs
        default:
            topProbs = nil
        }
        for _, tp := range topProbs {
            var tl float64
            switch {
            case tp.Logprob != nil:
                tl = *tp.Logprob
            case tp.Prob != nil:
                tl = *tp.Prob
            }
            result[i].TopLogprobs = append(result[i].TopLogprobs, TokenLogprob{Token: tp.Token, Logprob: tl})
        }
    }
    return result
}

func (s *llamaServerRunner) ApplyChatTemplate(ctx context.Context, req ChatRequest) (string, error) {
	data, err := s.llamaServerChatRequest(req, false)
	if err != nil {
		return "", err
	}

	body, err := json.Marshal(data)
	if err != nil {
		return "", fmt.Errorf("failed to marshal chat template request: %v", err)
	}

	endpoint := fmt.Sprintf("http://127.0.0.1:%d/apply-template", s.port)
	serverReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("error creating chat template request: %v", err)
	}
	serverReq.Header.Set("Content-Type", "application/json")

	res, err := s.httpClient().Do(serverReq)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return "", err
		}
		return "", fmt.Errorf("llama-server apply-template error: %w", err)
	}
	defer res.Body.Close()

	bodyBytes, err := io.ReadAll(res.Body)
	if err != nil {
		return "", fmt.Errorf("failed reading llama-server template response: %w", err)
	}
	if res.StatusCode >= 400 {
		return "", api.StatusError{StatusCode: res.StatusCode, ErrorMessage: s.statusErrorMessage(bodyBytes)}
	}

	var lsResp llamaServerApplyTemplateResponse
	if err := json.Unmarshal(bodyBytes, &lsResp); err != nil {
		return "", fmt.Errorf("error unmarshalling llama-server template response: %v", err)
	}
	if lsResp.Error != nil {
		return "", fmt.Errorf("llama-server template error: %v", lsResp.Error)
	}

	return lsResp.Prompt, nil
}

func (s *llamaServerRunner) Chat(ctx context.Context, req ChatRequest, fn func(ChatResponse)) error {
	slog.Debug("llama-server chat request", "messages", len(req.Messages))

	if req.Options == nil {
		opts := api.DefaultOptions()
		req.Options = &opts
	}

	if err := s.sem.Acquire(ctx, 1); err != nil {
		if errors.Is(err, context.Canceled) {
			slog.Info("aborting chat request due to client closing the connection")
		}
		return err
	}
	defer s.sem.Release(1)

	req.Options.NumPredict = -1

	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return err
	} else if status != ServerStatusReady {
		return fmt.Errorf("unexpected server status: %s", status)
	}

	lsReq, err := s.llamaServerChatRequest(req, true)
	if err != nil {
		return err
	}

	buffer := &bytes.Buffer{}
	enc := json.NewEncoder(buffer)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(lsReq); err != nil {
		return fmt.Errorf("failed to marshal chat request: %v", err)
	}

	endpoint := fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", s.port)
	serverReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, buffer)
	if err != nil {
		return fmt.Errorf("error creating chat request: %v", err)
	}
	serverReq.Header.Set("Content-Type", "application/json")

	res, err := s.httpClient().Do(serverReq)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return err
		}
		slog.Error("llama-server chat error", "error", err)
		if msg := s.lastErrMsg(); msg != "" {
			return fmt.Errorf("model runner has unexpectedly stopped, this may be due to resource limitations or an internal error, check yollama server logs for details: %s", msg)
		}
		return errors.New("model runner has unexpectedly stopped, this may be due to resource limitations or an internal error, check yollama server logs for details")
	}
	defer res.Body.Close()

	if res.StatusCode >= 400 {
		bodyBytes, err := io.ReadAll(res.Body)
		if err != nil {
			return fmt.Errorf("failed reading llama-server error response: %w", err)
		}

		return api.StatusError{StatusCode: res.StatusCode, ErrorMessage: s.statusErrorMessage(bodyBytes)}
	}

	scanner := bufio.NewScanner(res.Body)
	buf := make([]byte, 0, llamaServerStreamInitialBufferSize)
	scanner.Buffer(buf, llamaServerStreamMaxBufferSize)

	var finalResp ChatResponse
	var hasFinalResp bool

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}
			if bytes.HasPrefix(line, []byte(":")) {
				continue
			}

			evt, ok := bytes.CutPrefix(line, []byte("data: "))
			if !ok {
				evt = line
			}
			if bytes.Equal(evt, []byte("[DONE]")) {
				continue
			}

			var lsResp llamaServerChatResponse
			if err := json.Unmarshal(evt, &lsResp); err != nil {
				return fmt.Errorf("error unmarshalling llama-server chat response: %v", err)
			}
			if lsResp.Error != nil {
				return fmt.Errorf("llama-server chat error: %v", lsResp.Error)
			}
			if len(lsResp.Choices) == 0 {
				continue
			}

			choice := lsResp.Choices[0]
			resp := ChatResponse{
				Message: api.Message{
					Role:     "assistant",
					Content:  choice.Delta.Content,
					Thinking: choice.Delta.ReasoningContent,
				},
				Logprobs: convertLogprobs(choice.Logprobs.Content, req.TopLogprobs > 0),
			}

			if choice.FinishReason != nil {
				doneReason := DoneReasonStop
				if *choice.FinishReason == "length" {
					doneReason = DoneReasonLength
				}

				resp.Done = true
				resp.DoneReason = doneReason
				resp.PromptEvalCount = lsResp.Timings.promptEvalCount()
				resp.PromptEvalDuration = time.Duration(lsResp.Timings.PromptMS * float64(time.Millisecond))
				resp.EvalCount = lsResp.Timings.PredictN
				resp.EvalDuration = time.Duration(lsResp.Timings.PredictMS * float64(time.Millisecond))
				finalResp = resp
				hasFinalResp = true
				break
			}

			if resp.Message.Content != "" || resp.Message.Thinking != "" || len(resp.Logprobs) > 0 {
				fn(resp)
			}
		}

		if hasFinalResp {
			break
		}
	}

	if hasFinalResp {
		for scanner.Scan() {
		}

		if err := scanner.Err(); err != nil {
			if err := llamaServerStreamLimitError("chat response", err); err != nil {
				return err
			}
			return fmt.Errorf("error reading llama-server chat response: %v", err)
		}

		if err := res.Body.Close(); err != nil {
			return fmt.Errorf("error closing llama-server chat response: %v", err)
		}

		fn(finalResp)
		return nil
	}

	if err := scanner.Err(); err != nil {
		if err := llamaServerStreamLimitError("chat response", err); err != nil {
			return err
		}
		if strings.Contains(err.Error(), "unexpected EOF") || strings.Contains(err.Error(), "forcibly closed") {
			s.Close()
			msg := s.lastErrMsg()
			if msg == "" {
				msg = err.Error()
			}
			return fmt.Errorf("an error was encountered while running the model: %s", msg)
		}
		return fmt.Errorf("error reading llama-server chat response: %v", err)
	}

	return nil
}

func (s *llamaServerRunner) llamaServerChatRequest(req ChatRequest, stream bool) (map[string]any, error) {
	if req.Options == nil {
		opts := api.DefaultOptions()
		req.Options = &opts
	}

	messages := make([]map[string]any, 0, len(req.Messages))
	for _, msg := range req.Messages {
		converted, err := llamaServerChatMessage(MessageFromAPI(msg))
		if err != nil {
			return nil, err
		}
		messages = append(messages, converted)
	}

	body := map[string]any{
		"messages":          messages,
		"stream":            stream,
		"cache_prompt":      false,
		"n_predict":         req.Options.NumPredict,
		"n_keep":            req.Options.NumKeep,
		"temperature":       req.Options.Temperature,
		"top_k":             req.Options.TopK,
		"top_p":             req.Options.TopP,
		"min_p":             req.Options.MinP,
		"stop":              req.Options.Stop,
		"repeat_penalty":    req.Options.RepeatPenalty,
		"repeat_last_n":     req.Options.RepeatLastN,
		"frequency_penalty": req.Options.FrequencyPenalty,
		"presence_penalty":  req.Options.PresencePenalty,
		"typical_p":         req.Options.TypicalP,
		"seed":              req.Options.Seed,
	}
	if req.Logprobs {
		body["logprobs"] = true
		body["top_logprobs"] = max(req.TopLogprobs, 1)
	}
	if req.Think != nil {
		body["chat_template_kwargs"] = map[string]any{
			"enable_thinking": req.Think.Bool(),
		}
	}
	if format, err := llamaServerChatResponseFormat(req.Format); err != nil {
		return nil, err
	} else if format != nil {
		body["response_format"] = format
	}

	return body, nil
}

func llamaServerChatMessage(msg Message) (map[string]any, error) {
	converted := map[string]any{
		"role": msg.Role,
	}

	if len(msg.Media) == 0 {
		converted["content"] = msg.Content
		return converted, nil
	}

	parts := make([]map[string]any, 0, len(msg.Media)+1)
	if msg.Content != "" {
		parts = append(parts, map[string]any{
			"type": "text",
			"text": msg.Content,
		})
	}
	for _, media := range msg.Media {
		parts = append(parts, llamaServerChatMediaPart(media))
	}
	converted["content"] = parts
	return converted, nil
}

func llamaServerChatMediaPart(media MediaData) map[string]any {
	encoded := base64.StdEncoding.EncodeToString(media.Data)
	if format, ok := AudioFormat(media.Data); ok {
		return map[string]any{
			"type": "input_audio",
			"input_audio": map[string]any{
				"data":   encoded,
				"format": format,
			},
		}
	}

	mime := http.DetectContentType(media.Data)
	if !strings.HasPrefix(mime, "image/") {
		mime = "image/jpeg"
	}
	return map[string]any{
		"type": "image_url",
		"image_url": map[string]any{
			"url": "data:" + mime + ";base64," + encoded,
		},
	}
}

func (s *llamaServerRunner) Embedding(ctx context.Context, input string) ([]float32, int, error) {
	if err := s.sem.Acquire(ctx, 1); err != nil {
		return nil, 0, err
	}
	defer s.sem.Release(1)

	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return nil, 0, err
	} else if status != ServerStatusReady {
		return nil, 0, fmt.Errorf("unexpected server status: %s", status)
	}

	// Use "input" field (not "content") to get the OAI-compatible response format
	// which includes tokens_evaluated for prompt token counting
	data, err := json.Marshal(map[string]string{"input": input})
	if err != nil {
		return nil, 0, fmt.Errorf("error marshaling embed data: %w", err)
	}

	// Use /v1/embeddings (OAI-compatible) to get tokens_evaluated in the response
	r, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/v1/embeddings", s.port), bytes.NewBuffer(data))
	if err != nil {
		return nil, 0, fmt.Errorf("error creating embed request: %w", err)
	}
	r.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient().Do(r)
	if err != nil {
		return nil, 0, fmt.Errorf("do embedding request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, 0, fmt.Errorf("error reading embed response: %w", err)
	}

	if resp.StatusCode >= 400 {
		statusCode, errMsg := normalizeEmbeddingError(resp.StatusCode, body)
		return nil, 0, api.StatusError{StatusCode: statusCode, ErrorMessage: errMsg}
	}

	// With "input" field, llama-server returns OAI-compatible format:
	//   {"data": [{"embedding": [0.1, ...], "tokens_evaluated": N}], "usage": {"prompt_tokens": N}}
	// With "content" field, it returns:
	//   [{"embedding": [[0.1, ...]], "index": 0}]
	var oaiResp struct {
		Data []struct {
			Embedding       json.RawMessage `json:"embedding"`
			TokensEvaluated int             `json:"tokens_evaluated"`
		} `json:"data"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &oaiResp); err == nil && len(oaiResp.Data) > 0 {
		var embedding []float32
		if err := json.Unmarshal(oaiResp.Data[0].Embedding, &embedding); err != nil {
			return nil, 0, fmt.Errorf("unmarshal embedding values: %w", err)
		}
		promptTokens := oaiResp.Usage.PromptTokens
		if promptTokens == 0 {
			promptTokens = oaiResp.Data[0].TokensEvaluated
		}
		return embedding, promptTokens, nil
	}

	// Fallback: non-OAI array format [{"embedding": [[0.1, ...]], "index": 0}]
	var results []struct {
		Embedding json.RawMessage `json:"embedding"`
	}
	if err := json.Unmarshal(body, &results); err != nil {
		return nil, 0, fmt.Errorf("unmarshal embedding response: %w", err)
	}
	if len(results) == 0 {
		return nil, 0, fmt.Errorf("empty embedding response")
	}

	var embedding []float32
	if err := json.Unmarshal(results[0].Embedding, &embedding); err != nil {
		var nested [][]float32
		if err2 := json.Unmarshal(results[0].Embedding, &nested); err2 != nil {
			return nil, 0, fmt.Errorf("unmarshal embedding values: %w (also tried nested: %w)", err, err2)
		}
		if len(nested) > 0 {
			embedding = nested[0]
		}
	}

	return embedding, 0, nil
}

func normalizeEmbeddingError(statusCode int, body []byte) (int, string) {
	raw := strings.TrimSpace(string(body))
	errMsg := extractLlamaServerErrorMessage(body)
	if errMsg == "" {
		errMsg = raw
	}

	if isEmbeddingInputLimitError(errMsg) || isEmbeddingInputLimitError(raw) {
		return http.StatusBadRequest, "the input length exceeds the context length"
	}

	return statusCode, errMsg
}

func llamaServerChatResponseFormat(format json.RawMessage) (map[string]any, error) {
	if len(format) == 0 {
		return nil, nil
	}

	switch string(format) {
	case `null`, `""`:
		return nil, nil
	case `"json"`:
		return map[string]any{"type": "json_object"}, nil
	default:
		if format[0] != '{' {
			return nil, fmt.Errorf("invalid format: %q; expected \"json\" or a valid JSON Schema object", format)
		}

		var schema map[string]any
		if err := json.Unmarshal(format, &schema); err != nil {
			return nil, fmt.Errorf("invalid format: %q; expected \"json\" or a valid JSON Schema object", format)
		}

		return map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   "schema",
				"schema": schema,
			},
		}, nil
	}
}

func extractLlamaServerErrorMessage(body []byte) string {
	var resp struct {
		Error json.RawMessage `json:"error"`
	}
	if err := json.Unmarshal(body, &resp); err != nil || len(resp.Error) == 0 {
		return ""
	}

	var msg string
	if err := json.Unmarshal(resp.Error, &msg); err == nil {
		return strings.TrimSpace(msg)
	}

	var nested struct {
		Message string `json:"message"`
	}
	if err := json.Unmarshal(resp.Error, &nested); err == nil {
		return strings.TrimSpace(nested.Message)
	}

	return ""
}

func isEmbeddingInputLimitError(errMsg string) bool {
	msg := strings.ToLower(errMsg)
	return strings.Contains(msg, "too large") ||
		strings.Contains(msg, "context size") ||
		strings.Contains(msg, "context length") ||
		strings.Contains(msg, "physical batch size") ||
		strings.Contains(msg, "exceeds the available context")
}

func (s *llamaServerRunner) tokenize(ctx context.Context, content any, addSpecial bool, parseSpecial *bool) ([]int, error) {
	req := struct {
		Content      any   `json:"content"`
		AddSpecial   bool  `json:"add_special,omitempty"`
		ParseSpecial *bool `json:"parse_special,omitempty"`
	}{
		Content:      content,
		AddSpecial:   addSpecial,
		ParseSpecial: parseSpecial,
	}

	data, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	r, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/tokenize", s.port), bytes.NewBuffer(data))
	if err != nil {
		return nil, err
	}
	r.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient().Do(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("tokenize error: %s", body)
	}

	var result struct {
		Tokens []int `json:"tokens"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, err
	}

	return result.Tokens, nil
}

// Tokenize calls llama-server's /tokenize endpoint.
func (s *llamaServerRunner) Tokenize(ctx context.Context, content string) ([]int, error) {
	return s.tokenize(ctx, content, false, nil)
}

// Detokenize calls llama-server's /detokenize endpoint.
func (s *llamaServerRunner) Detokenize(ctx context.Context, tokens []int) (string, error) {
	data, err := json.Marshal(map[string][]int{"tokens": tokens})
	if err != nil {
		return "", err
	}

	r, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/detokenize", s.port), bytes.NewBuffer(data))
	if err != nil {
		return "", err
	}
	r.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient().Do(r)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	if resp.StatusCode >= 400 {
		return "", fmt.Errorf("detokenize error: %s", body)
	}

	var result struct {
		Content string `json:"content"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return "", err
	}

	return result.Content, nil
}

func (s *llamaServerRunner) Close() error {
	return s.stopProcess()
}

func (s *llamaServerRunner) stopProcess() error {
	if s.cmd != nil && s.cmd.Process != nil {
		if s.cmd.ProcessState != nil {
			return nil
		}
		slog.Debug("stopping llama-server", "pid", s.Pid())
		if err := s.cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
			return err
		}
		if s.done != nil {
			slog.Debug("waiting for llama-server to exit", "pid", s.Pid())
			<-s.done
		}
		slog.Debug("llama-server stopped", "pid", s.Pid())
	}
	return nil
}

// GetDeviceInfos returns device info for GPUs used by this runner, with FreeMemory
// updated to reflect actual usage. Uses the minimum of:
//   - Our accounting: TotalMemory minus tracked VRAM allocations
//   - System-reported: free VRAM from llama-server at load time minus our allocations
//
// The min-of-two approach handles both our own usage (accurate) and external
// consumers (system-reported, may be optimistic on some platforms).
func (s *llamaServerRunner) GetDeviceInfos(ctx context.Context) []ml.DeviceInfo {
	if len(s.gpus) == 0 {
		return nil
	}
	s.memoryMu.RLock()
	defer s.memoryMu.RUnlock()

	infos := make([]ml.DeviceInfo, len(s.gpus))
	for i, gpu := range s.gpus {
		infos[i] = gpu
		used := s.vramByDevice[gpu.Name]

		// Our accounting: total minus what we allocated
		var accountedFree uint64
		if used < gpu.TotalMemory {
			accountedFree = gpu.TotalMemory - used
		}

		// System-reported: what the GPU said was free at load time, minus what
		// we've allocated since. This captures external consumers on platforms
		// where the driver reports accurately.
		systemFree := accountedFree // default to our accounting
		if sysFree, ok := s.systemFreeAtLoad[gpu.Name]; ok {
			if used < sysFree {
				systemFree = sysFree - used
			} else {
				systemFree = 0
			}
		}

		// Take the minimum — never optimistic
		infos[i].FreeMemory = min(accountedFree, systemFree)
	}
	return infos
}

// MemorySize returns total and GPU memory usage parsed from llama-server's
// post-load log output. Full model-layer offload is reported as 100% GPU.
func (s *llamaServerRunner) MemorySize() (total, vram uint64) {
	s.memoryMu.RLock()
	memTotal := s.memTotal
	memGPU := s.memGPU
	memModelFileBacked := s.memModelFileBacked
	memCPUMappedModel := s.memCPUMappedModel
	totalLayers := s.totalLayers
	gpuLayers := s.gpuLayers
	gpuLayerOverflow := s.gpuLayerOverflow
	s.memoryMu.RUnlock()

	if memTotal > 0 {
		total, vram = memTotal, memGPU
		// With mmap, llama-server reports each CPU_Mapped model buffer as the
		// file-offset span of its CPU-resident tensors. During partial offload
		// that span covers nearly the whole file (the first and last tensors
		// stay on CPU), re-counting weights already held in device buffers.
		// Only buffers that mirror the on-disk file can overlap this way;
		// repacked copies such as CPU_REPACK are separate real allocations and
		// must be left intact. Weights cannot exceed the model file on disk, so
		// trim that overlap from the mmap-backed (reclaimable page cache) portion.
		if memCPUMappedModel > 0 {
			if info, err := os.Stat(s.modelPath); err == nil && memModelFileBacked > uint64(info.Size()) {
				total -= min(memCPUMappedModel, memModelFileBacked-uint64(info.Size()))
			}
		}
		if totalLayers > 0 && gpuLayers >= totalLayers && gpuLayerOverflow == 0 {
			total = vram
		}
		return total, vram
	}
	
	// Fallback: use model file size as a rough proxy
	slog.Debug("llama-server buffer sizes not available, falling back to file size estimate", "model", s.modelPath)
	if info, err := os.Stat(s.modelPath); err == nil {
		total = uint64(info.Size())
		vram = total
	}

	return total, vram
}

// memoryParsingWriter wraps an io.Writer and parses llama-server log output
// for buffer size lines. It updates the runner's per-device VRAM tracking.
//
// Parsed line formats (all backends):
//
//	CUDA0 model buffer size =   852.89 MiB
//	CUDA0 KV buffer size =  1920.00 MiB
//	CUDA0 compute buffer size =   378.04 MiB
//	CPU_Mapped model buffer size =   308.23 MiB
//	CUDA_Host compute buffer size =   268.05 MiB
//	MTL0_Mapped model buffer size =  1918.35 MiB
//	ROCm0 model buffer size =  1918.35 MiB
type memoryParsingWriter struct {
	inner   io.Writer
	runner  *llamaServerRunner
	buffers map[memoryBufferKey]memoryBuffer
}

type memoryBufferKey struct {
	component string
	backend   string
	kind      string
}

type memoryBuffer struct {
	bytes uint64
}

// deviceFreeRegex matches per-device free VRAM reported at model load time:
//
//	using device CUDA0 (NVIDIA GeForce RTX 4060 Ti) (0000:01:00.0) - 15221 MiB free
//	using device MTL0 (Apple M5 Max) (unknown id) - 110100 MiB free
//	using device ROCm0 (AMD Radeon RX 6800) (0000:06:00.0) - 16196 MiB free
var deviceFreeRegex = regexp.MustCompile(`using device (\S+)\s+\(.*\)\s+-\s+(\d+)\s+MiB free`)

// bufferSizeRegex matches llama-server buffer size lines and captures the
// component so repeated fit/probe values can be replaced by the final load.
var bufferSizeRegex = regexp.MustCompile(`(?m)(?:^|\n)[^\n:]*?([A-Za-z_][A-Za-z0-9_]*):\s+(\S+)\s+(model|KV|compute|output|RS)\s+buffer size\s*=\s*([\d.]+)\s*MiB`)

var (
	offloadedLayersRegex      = regexp.MustCompile(`offloaded\s+(\d+)/(\d+)\s+layers to GPU`)
	fitOverflowingLayersRegex = regexp.MustCompile(`common_params_fit_impl:\s+-\s+.+:\s+\d+\s+layers\s+\(\s*(\d+)\s+overflowing\)`)
)

// isGPUBuffer returns true if the backend buffer name represents GPU memory.
// CPU, BLAS, and host-pinned buffers (*_Host) are not GPU memory.
// Device-mapped buffers (e.g., MTL0_Mapped) ARE GPU memory — they're model
// weights in device-accessible memory. Only CPU_Mapped is CPU memory.
func isGPUBuffer(name string) bool {
	if name == "CPU" || name == "BLAS" || strings.HasPrefix(name, "CPU_") {
		return false
	}
	if strings.HasSuffix(name, "_Host") {
		return false
	}
	return true
}

// deviceName returns the base device name for per-device VRAM tracking.
// Strips suffixes like _Mapped, _REPACK so that e.g. "MTL0_Mapped" is
// tracked under "MTL0" alongside "MTL0 KV buffer" and "MTL0 compute buffer".
func deviceName(backendName string) string {
	for _, suffix := range []string{"_Mapped", "_REPACK", "_Private"} {
		if strings.HasSuffix(backendName, suffix) {
			return strings.TrimSuffix(backendName, suffix)
		}
	}
	return backendName
}

func (w *memoryParsingWriter) Write(b []byte) (int, error) {
	if w.runner != nil {
		func() {
			w.runner.memoryMu.Lock()
			defer w.runner.memoryMu.Unlock()

			if match := deviceFreeRegex.FindSubmatch(b); match != nil {
				devName := string(match[1])
				if mib, err := strconv.ParseUint(string(match[2]), 10, 64); err == nil {
					w.runner.systemFreeAtLoad[devName] = mib * 1024 * 1024
				}
			}
			for _, match := range offloadedLayersRegex.FindAllSubmatch(b, -1) {
				loaded, loadedErr := strconv.ParseUint(string(match[1]), 10, 64)
				total, totalErr := strconv.ParseUint(string(match[2]), 10, 64)
				if loadedErr == nil && totalErr == nil {
					w.runner.gpuLayers = loaded
					w.runner.totalLayers = total
				}
			}
			for _, match := range fitOverflowingLayersRegex.FindAllSubmatch(b, -1) {
				overflowing, err := strconv.ParseUint(string(match[1]), 10, 64)
				if err == nil && overflowing > 0 {
					w.runner.gpuLayerOverflow += int(overflowing)
				}
			}
			for _, match := range bufferSizeRegex.FindAllSubmatch(b, -1) {
				backendName := string(match[2])
				if mib, err := strconv.ParseFloat(string(match[4]), 64); err == nil {
					if w.buffers == nil {
						w.buffers = make(map[memoryBufferKey]memoryBuffer)
					}
					w.buffers[memoryBufferKey{
						component: string(match[1]),
						backend:   backendName,
						kind:      string(match[3]),
					}] = memoryBuffer{bytes: uint64(mib * 1024 * 1024)}
					w.updateRunnerMemoryLocked()
				}
			}
		}()
	}
	return w.inner.Write(b)
}

func (w *memoryParsingWriter) updateRunnerMemoryLocked() {
	var total, gpu, modelFileBacked, cpuMappedModel uint64
	byDevice := make(map[string]uint64)

	for key, buffer := range w.buffers {
		total += buffer.bytes
		if key.kind == "model" {
			onGPU := isGPUBuffer(key.backend)
			mmapBacked := strings.HasSuffix(key.backend, "_Mapped")
			// Device copies and mmap views mirror the on-disk weights, so their
			// spans can overlap and double-count on partial offload. Repacked or
			// host-pinned CPU copies (e.g. CPU_REPACK) are separate real
			// allocations that never overlap the file, so keep them out of the base.
			if onGPU || mmapBacked {
				modelFileBacked += buffer.bytes
			}
			if !onGPU && mmapBacked {
				cpuMappedModel += buffer.bytes
			}
		}
		if isGPUBuffer(key.backend) {
			gpu += buffer.bytes
			byDevice[deviceName(key.backend)] += buffer.bytes
		}
	}

	w.runner.memTotal = total
	w.runner.memGPU = gpu
	w.runner.memModelFileBacked = modelFileBacked
	w.runner.memCPUMappedModel = cpuMappedModel
	w.runner.vramByDevice = byDevice
}

// VRAMByGPU returns the VRAM used by this runner on the specified device.
// The values are parsed from llama-server's buffer size log output during model load
// (model tensors + KV cache + compute buffers).
func (s *llamaServerRunner) VRAMByGPU(id ml.DeviceID) uint64 {
	s.memoryMu.RLock()
	defer s.memoryMu.RUnlock()

	// Map DeviceID to the log device name used by llama-server.
	// Discovery stores the device name (e.g., "CUDA0", "ROCm0", "MTL0") from
	// --list-devices stdout, which matches the buffer log prefix.
	for _, gpu := range s.gpus {
		if gpu.DeviceID == id {
			return s.vramByDevice[gpu.Name]
		}
	}
	return 0
}
