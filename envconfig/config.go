package envconfig

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Host returns the scheme and host. Host can be configured via the YOLLAMA_HOST environment variable.
// Default is scheme "http" and host "127.0.0.1:11434"
func Host() *url.URL {
	defaultPort := "11434"

	s := strings.TrimSpace(Var("YOLLAMA_HOST"))
	scheme, hostport, ok := strings.Cut(s, "://")
	switch {
	case !ok:
		scheme, hostport = "http", s
	case scheme == "http":
		defaultPort = "11434"
	case scheme == "https":
		defaultPort = "11434"
	}

	hostport, path, _ := strings.Cut(hostport, "/")
	host, port, err := net.SplitHostPort(hostport)
	if err != nil {
		host, port = "127.0.0.1", defaultPort
		if ip := net.ParseIP(strings.Trim(hostport, "[]")); ip != nil {
			host = ip.String()
		} else if hostport != "" {
			host = hostport
		}
	}

	if n, err := strconv.ParseInt(port, 10, 32); err != nil || n > 65535 || n < 0 {
		slog.Warn("invalid port, using default", "port", port, "default", defaultPort)
		port = defaultPort
	}

	return &url.URL{
		Scheme: scheme,
		Host:   net.JoinHostPort(host, port),
		Path:   path,
	}
}

// ConnectableHost returns Host() with unspecified bind addresses (0.0.0.0, ::)
// replaced by the corresponding loopback address (127.0.0.1, ::1).
// Unspecified addresses are valid for binding a server socket but not for
// connecting as a client, which fails on Windows.
func ConnectableHost() *url.URL {
	u := Host()
	host, port, err := net.SplitHostPort(u.Host)
	if err != nil {
		return u
	}

	if ip := net.ParseIP(host); ip != nil && ip.IsUnspecified() {
		if ip.To4() != nil {
			host = "127.0.0.1"
		} else {
			host = "::1"
		}
		u.Host = net.JoinHostPort(host, port)
	}

	return u
}

// AllowedOrigins returns a list of allowed origins. AllowedOrigins can be configured via the YOLLAMA_ORIGINS environment variable.
func AllowedOrigins() (origins []string) {
	if s := Var("YOLLAMA_ORIGINS"); s != "" {
		origins = strings.Split(s, ",")
	}

	for _, origin := range []string{"localhost", "127.0.0.1", "0.0.0.0"} {
		origins = append(origins,
			fmt.Sprintf("http://%s", origin),
			fmt.Sprintf("https://%s", origin),
			fmt.Sprintf("http://%s", net.JoinHostPort(origin, "*")),
			fmt.Sprintf("https://%s", net.JoinHostPort(origin, "*")),
		)
	}

	origins = append(origins,
		"app://*",
		"file://*",
		"tauri://*",
		"vscode-webview://*",
		"vscode-file://*",
	)

	return origins
}

// Models returns the path to the models directory. Models directory can be configured via the YOLLAMA_MODELS environment variable.
// Default is $HOME/.yollama/models
func Models() string {
	if s := Var("YOLLAMA_MODELS"); s != "" {
		return s
	}

	home, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}

	return filepath.Join(home, ".yollama", "models")
}

// KeepAlive returns the duration that models stay loaded in memory. KeepAlive can be configured via the YOLLAMA_KEEP_ALIVE environment variable.
// Negative values are treated as infinite. Zero is treated as no keep alive.
// Default is 5 minutes.
func KeepAlive() (keepAlive time.Duration) {
	keepAlive = 5 * time.Minute
	if s := Var("YOLLAMA_KEEP_ALIVE"); s != "" {
		if d, err := time.ParseDuration(s); err == nil {
			keepAlive = d
		} else if n, err := strconv.ParseInt(s, 10, 64); err == nil {
			keepAlive = time.Duration(n) * time.Second
		}
	}

	if keepAlive < 0 {
		return time.Duration(math.MaxInt64)
	}

	return keepAlive
}

// LoadTimeout returns the duration for stall detection during model loads. LoadTimeout can be configured via the YOLLAMA_LOAD_TIMEOUT environment variable.
// Zero or Negative values are treated as infinite.
// Default is 5 minutes.
func LoadTimeout() (loadTimeout time.Duration) {
	loadTimeout = 5 * time.Minute
	if s := Var("YOLLAMA_LOAD_TIMEOUT"); s != "" {
		if d, err := time.ParseDuration(s); err == nil {
			loadTimeout = d
		} else if n, err := strconv.ParseInt(s, 10, 64); err == nil {
			loadTimeout = time.Duration(n) * time.Second
		}
	}

	if loadTimeout <= 0 {
		return time.Duration(math.MaxInt64)
	}

	return loadTimeout
}

func Remotes() []string {
	var r []string
	raw := strings.TrimSpace(Var("YOLLAMA_REMOTES"))
	if raw == "" {
		r = []string{"*.*.*.*"}
	} else {
		r = strings.Split(raw, ",")
	}
	return r
}

func BoolWithDefault(k string) func(defaultValue bool) bool {
	return func(defaultValue bool) bool {
		if s := Var(k); s != "" {
			b, err := strconv.ParseBool(s)
			if err != nil {
				return true
			}

			return b
		}

		return defaultValue
	}
}

func Bool(k string) func() bool {
	withDefault := BoolWithDefault(k)
	return func() bool {
		return withDefault(false)
	}
}

// LogLevel returns the log level for the application.
// Values are 0 or false INFO (Default), 1 or true DEBUG, 2 TRACE
func LogLevel() slog.Level {
	level := slog.LevelInfo
	if s := Var("YOLLAMA_DEBUG"); s != "" {
		if b, _ := strconv.ParseBool(s); b {
			level = slog.LevelDebug
		} else if i, _ := strconv.ParseInt(s, 10, 64); i != 0 {
			level = slog.Level(i * -4)
		}
	}

	return level
}

var (
	// FlashAttention enables the experimental flash attention feature.
	FlashAttention = BoolWithDefault("YOLLAMA_FLASH_ATTENTION")
	// GoTemplate enables Modelfile TEMPLATE rendering when a model has one.
	GoTemplate = BoolWithDefault("YOLLAMA_GO_TEMPLATE")
	// DebugLogRequests logs inference requests to disk for replay/debugging.
	DebugLogRequests = Bool("YOLLAMA_DEBUG_LOG_REQUESTS")
	// KvCacheType is the quantization type for the K/V cache.
	KvCacheType = String("YOLLAMA_KV_CACHE_TYPE")
	// NoHistory disables readline history.
	NoHistory = Bool("YOLLAMA_NOHISTORY")
	// NoPrune disables pruning of model blobs on startup.
	NoPrune = Bool("YOLLAMA_NOPRUNE")
	// SchedSpread allows scheduling models across all GPUs.
	SchedSpread = Bool("YOLLAMA_SCHED_SPREAD")
	// ContextLength sets the default context length
	ContextLength = Uint("YOLLAMA_CONTEXT_LENGTH", 0)
	// Auth enables authentication between the Yollama client and server
	UseAuth = Bool("YOLLAMA_AUTH")
	// EnableVulkan controls Vulkan backend discovery.
	EnableVulkan = BoolWithDefault("YOLLAMA_VULKAN")
	// EnableIntegratedGPU controls whether integrated GPUs may be selected.
	EnableIntegratedGPU = BoolWithDefault("YOLLAMA_IGPU_ENABLE")
	// NoCloudEnv checks the YOLLAMA_NO_CLOUD environment variable.
	NoCloudEnv = Bool("YOLLAMA_NO_CLOUD")
)

func String(s string) func() string {
	return func() string {
		return Var(s)
	}
}

var (
	LLMLibrary = String("YOLLAMA_LLM_LIBRARY")
	Editor     = String("YOLLAMA_EDITOR")

	CudaVisibleDevices    = String("CUDA_VISIBLE_DEVICES")
	HipVisibleDevices     = String("HIP_VISIBLE_DEVICES")
	RocrVisibleDevices    = String("ROCR_VISIBLE_DEVICES")
	VkVisibleDevices      = String("GGML_VK_VISIBLE_DEVICES")
	GpuDeviceOrdinal      = String("GPU_DEVICE_ORDINAL")
	HsaOverrideGfxVersion = String("HSA_OVERRIDE_GFX_VERSION")
)

func Uint(key string, defaultValue uint) func() uint {
	return func() uint {
		if s := Var(key); s != "" {
			if n, err := strconv.ParseUint(s, 10, 64); err != nil {
				slog.Warn("invalid environment variable, using default", "key", key, "value", s, "default", defaultValue)
			} else {
				return uint(n)
			}
		}

		return defaultValue
	}
}

var (
	// NumParallel sets the number of parallel model requests. NumParallel can be configured via the YOLLAMA_NUM_PARALLEL environment variable.
	NumParallel = Uint("YOLLAMA_NUM_PARALLEL", 1)
	// MaxRunners sets the maximum number of loaded models. MaxRunners can be configured via the YOLLAMA_MAX_LOADED_MODELS environment variable.
	MaxRunners = Uint("YOLLAMA_MAX_LOADED_MODELS", 1)
	// MaxQueue sets the maximum number of queued requests. MaxQueue can be configured via the YOLLAMA_MAX_QUEUE environment variable.
	MaxQueue = Uint("YOLLAMA_MAX_QUEUE", 512)
	// MaxTransferStreams caps the number of simultaneous body-bearing
	// transfers during safetensors model pulls/pushes, keeping slower
	// networks from being saturated. Tune higher for fast networks. Has
	// no effect on GGUF transfers, which use the legacy upload/download
	// paths.
	MaxTransferStreams = Uint("YOLLAMA_MAX_TRANSFER_STREAMS", 4)
)

func Uint64(key string, defaultValue uint64) func() uint64 {
	return func() uint64 {
		if s := Var(key); s != "" {
			if n, err := strconv.ParseUint(s, 10, 64); err != nil {
				slog.Warn("invalid environment variable, using default", "key", key, "value", s, "default", defaultValue)
			} else {
				return n
			}
		}

		return defaultValue
	}
}

// Set aside VRAM per GPU
var GpuOverhead = Uint64("YOLLAMA_GPU_OVERHEAD", 0)

type EnvVar struct {
	Name        string
	Value       any
	Description string
}

func AsMap() map[string]EnvVar {
	ret := map[string]EnvVar{
		"YOLLAMA_DEBUG":                {"YOLLAMA_DEBUG", LogLevel(), "Show additional debug information (e.g. YOLLAMA_DEBUG=1)"},
		"YOLLAMA_DEBUG_LOG_REQUESTS":   {"YOLLAMA_DEBUG_LOG_REQUESTS", DebugLogRequests(), "Log inference request bodies and replay curl commands to a temp directory"},
		"YOLLAMA_GO_TEMPLATE":          {"YOLLAMA_GO_TEMPLATE", GoTemplate(true), "Enable Modelfile TEMPLATE based rendering when available"},
		"YOLLAMA_FLASH_ATTENTION":      {"YOLLAMA_FLASH_ATTENTION", FlashAttention(false), "Enabled flash attention"},
		"YOLLAMA_KV_CACHE_TYPE":        {"YOLLAMA_KV_CACHE_TYPE", KvCacheType(), "Quantization type for the K/V cache (default: f16)"},
		"YOLLAMA_GPU_OVERHEAD":         {"YOLLAMA_GPU_OVERHEAD", GpuOverhead(), "Reserve a portion of VRAM per GPU (bytes)"},
		"YOLLAMA_IGPU_ENABLE":          {"YOLLAMA_IGPU_ENABLE", String("YOLLAMA_IGPU_ENABLE")(), "Enable integrated GPUs"},
		"LLAMA_ARG_FIT":               {"LLAMA_ARG_FIT", String("LLAMA_ARG_FIT")(), "Enable llama.cpp automatic fit of unset memory options (default \"on\")"},
		"LLAMA_ARG_FIT_TARGET":        {"LLAMA_ARG_FIT_TARGET", String("LLAMA_ARG_FIT_TARGET")(), "Target free VRAM margin per device for llama.cpp fit (MiB)"},
		"YOLLAMA_HOST":                 {"YOLLAMA_HOST", Host(), "IP Address for the yollama server (default 127.0.0.1:11434)"},
		"YOLLAMA_KEEP_ALIVE":           {"YOLLAMA_KEEP_ALIVE", KeepAlive(), "The duration that models stay loaded in memory (default \"5m\")"},
		"YOLLAMA_LLM_LIBRARY":          {"YOLLAMA_LLM_LIBRARY", LLMLibrary(), "Set LLM library to bypass autodetection"},
		"YOLLAMA_LOAD_TIMEOUT":         {"YOLLAMA_LOAD_TIMEOUT", LoadTimeout(), "How long to allow model loads to stall before giving up (default \"5m\")"},
		"YOLLAMA_MAX_LOADED_MODELS":    {"YOLLAMA_MAX_LOADED_MODELS", MaxRunners(), "Maximum number of loaded models per GPU"},
		"YOLLAMA_MAX_TRANSFER_STREAMS": {"YOLLAMA_MAX_TRANSFER_STREAMS", MaxTransferStreams(), "Maximum parallel transfer streams for safetensors model pulls/pushes (default 4)"},
		"YOLLAMA_MAX_QUEUE":            {"YOLLAMA_MAX_QUEUE", MaxQueue(), "Maximum number of queued requests"},
		"YOLLAMA_MODELS":               {"YOLLAMA_MODELS", Models(), "The path to the models directory"},
		"YOLLAMA_NO_CLOUD":             {"YOLLAMA_NO_CLOUD", NoCloud(), "Disable Yollama cloud features (remote inference and web search)"},
		"YOLLAMA_NOHISTORY":            {"YOLLAMA_NOHISTORY", NoHistory(), "Do not preserve readline history"},
		"YOLLAMA_NOPRUNE":              {"YOLLAMA_NOPRUNE", NoPrune(), "Do not prune model blobs on startup"},
		"YOLLAMA_NUM_PARALLEL":         {"YOLLAMA_NUM_PARALLEL", NumParallel(), "Maximum number of parallel requests"},
		"YOLLAMA_ORIGINS":              {"YOLLAMA_ORIGINS", AllowedOrigins(), "A comma separated list of allowed origins"},
		"YOLLAMA_SCHED_SPREAD":         {"YOLLAMA_SCHED_SPREAD", SchedSpread(), "Always schedule model across all GPUs"},
		"YOLLAMA_CONTEXT_LENGTH":       {"YOLLAMA_CONTEXT_LENGTH", ContextLength(), "Context length to use unless otherwise specified (default: 4k/32k/256k based on VRAM)"},
		"YOLLAMA_EDITOR":               {"YOLLAMA_EDITOR", Editor(), "Path to editor for interactive prompt editing (Ctrl+G)"},
		"YOLLAMA_REMOTES":              {"YOLLAMA_REMOTES", Remotes(), "Allowed hosts for remote models (default \"127.0.0.1\")"},

		// Informational
		"HTTP_PROXY":  {"HTTP_PROXY", String("HTTP_PROXY")(), "HTTP proxy"},
		"HTTPS_PROXY": {"HTTPS_PROXY", String("HTTPS_PROXY")(), "HTTPS proxy"},
		"NO_PROXY":    {"NO_PROXY", String("NO_PROXY")(), "No proxy"},
	}

	if runtime.GOOS != "windows" {
		// Windows environment variables are case-insensitive so there's no need to duplicate them
		ret["http_proxy"] = EnvVar{"http_proxy", String("http_proxy")(), "HTTP proxy"}
		ret["https_proxy"] = EnvVar{"https_proxy", String("https_proxy")(), "HTTPS proxy"}
		ret["no_proxy"] = EnvVar{"no_proxy", String("no_proxy")(), "No proxy"}
	}

	if runtime.GOOS != "darwin" {
		ret["CUDA_VISIBLE_DEVICES"] = EnvVar{"CUDA_VISIBLE_DEVICES", CudaVisibleDevices(), "Set which NVIDIA devices are visible"}
		ret["HIP_VISIBLE_DEVICES"] = EnvVar{"HIP_VISIBLE_DEVICES", HipVisibleDevices(), "Set which AMD devices are visible by numeric ID"}
		ret["ROCR_VISIBLE_DEVICES"] = EnvVar{"ROCR_VISIBLE_DEVICES", RocrVisibleDevices(), "Set which AMD devices are visible by UUID or numeric ID"}
		ret["GGML_VK_VISIBLE_DEVICES"] = EnvVar{"GGML_VK_VISIBLE_DEVICES", VkVisibleDevices(), "Set which Vulkan devices are visible by numeric ID"}
		ret["GPU_DEVICE_ORDINAL"] = EnvVar{"GPU_DEVICE_ORDINAL", GpuDeviceOrdinal(), "Set which AMD devices are visible by numeric ID"}
		ret["HSA_OVERRIDE_GFX_VERSION"] = EnvVar{"HSA_OVERRIDE_GFX_VERSION", HsaOverrideGfxVersion(), "Override the gfx used for all detected AMD GPUs"}
		ret["YOLLAMA_VULKAN"] = EnvVar{"YOLLAMA_VULKAN", EnableVulkan(true), "Enable Vulkan support"}
	}

	return ret
}

func Values() map[string]string {
	vals := make(map[string]string)
	for k, v := range AsMap() {
		vals[k] = fmt.Sprintf("%v", v.Value)
	}
	return vals
}

// Var returns an environment variable stripped of leading and trailing quotes or spaces
func Var(key string) string {
	return strings.Trim(strings.TrimSpace(os.Getenv(key)), "\"'")
}

// serverConfigData holds the parsed fields from ~/.yollama/server.json.
type serverConfigData struct {
	DisableYollamaCloud bool `json:"disable_yollama_cloud,omitempty"`
}

var (
	serverCfgMu     sync.RWMutex
	serverCfgLoaded bool
	serverCfg       serverConfigData
)

func loadServerConfig() {
	serverCfgMu.RLock()
	if serverCfgLoaded {
		serverCfgMu.RUnlock()
		return
	}
	serverCfgMu.RUnlock()

	cfg := serverConfigData{}
	home, err := os.UserHomeDir()
	if err == nil {
		path := filepath.Join(home, ".yollama", "server.json")
		data, err := os.ReadFile(path)
		if err != nil {
			if !errors.Is(err, os.ErrNotExist) {
				slog.Debug("envconfig: could not read server config", "error", err)
			}
		} else if err := json.Unmarshal(data, &cfg); err != nil {
			slog.Debug("envconfig: could not parse server config", "error", err)
		}
	}

	serverCfgMu.Lock()
	defer serverCfgMu.Unlock()
	if serverCfgLoaded {
		return
	}
	serverCfg = cfg
	serverCfgLoaded = true
}

func cachedServerConfig() serverConfigData {
	serverCfgMu.RLock()
	defer serverCfgMu.RUnlock()
	return serverCfg
}

// ReloadServerConfig refreshes the cached ~/.yollama/server.json settings.
func ReloadServerConfig() {
	serverCfgMu.Lock()
	serverCfgLoaded = false
	serverCfg = serverConfigData{}
	serverCfgMu.Unlock()
	loadServerConfig()
}

// NoCloud returns true if Yollama cloud features are disabled,
// checking both the YOLLAMA_NO_CLOUD environment variable and
// the disable_yollama_cloud field in ~/.yollama/server.json.
func NoCloud() bool {
	return true
}

// NoCloudSource returns the source of the cloud-disabled decision.
// Returns "none", "env", "config", or "both".
func NoCloudSource() string {
	envDisabled := NoCloudEnv()
	loadServerConfig()
	configDisabled := cachedServerConfig().DisableYollamaCloud

	switch {
	case envDisabled && configDisabled:
		return "both"
	case envDisabled:
		return "env"
	case configDisabled:
		return "config"
	default:
		return "none"
	}
}
