package hardware

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"unsafe"
)

/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>
#include <stdlib.h>
*/
import "C"

// DynamicEngine wraps the loaded CGO library handle for runtime execution
type DynamicEngine struct {
	Handle   unsafe.Pointer
	FilePath string
	Mode     AccelerationType
}

// LoadOptimalEngine dynamically discovers and opens the best microarchitecture/CUDA backend
func LoadOptimalEngine(libDir string) (*DynamicEngine, error) {
	probe := DetectOptimalBackend(libDir)
	
	cPath := C.CString(probe.LibraryPath)
	defer C.free(unsafe.Pointer(cPath))

	// RTLD_LAZY | RTLD_GLOBAL allows symbols to resolve across companion ggml libraries
	handle := C.dlopen(cPath, C.RTLD_LAZY|C.RTLD_GLOBAL)
	if handle == nil {
		errStr := C.GoString(C.dlerror())
		return nil, fmt.Errorf("failed to dlopen %s: %s", probe.LibraryPath, errStr)
	}

	return &DynamicEngine{
		Handle:   handle,
		FilePath: probe.LibraryPath,
		Mode:     probe.Mode,
	}, nil
}

// LookupSymbol resolves a specific function pointer from the dynamically loaded backend (e.g. "llama_print_system_info")
func (e *DynamicEngine, symbol string) (unsafe.Pointer, error) {
	cSym := C.CString(symbol)
	defer C.free(unsafe.Pointer(cSym))

	// Clear existing errors
	C.dlerror()
	sym := C.dlsym(e.Handle, cSym)
	if errStr := C.dlerror(); errStr != nil {
		return nil, fmt.Errorf("failed to resolve symbol %s: %s", symbol, C.GoString(errStr))
	}

	return sym, nil
}

// Close unloads the runtime driver cleanly
func (e *DynamicEngine) Close() error {
	if e.Handle != nil {
		if C.dlclose(e.Handle) != 0 {
			return fmt.Errorf("failed to close dynamic library handle: %s", C.GoString(C.dlerror()))
		}
	}
	return nil
}

// Reuse or reference the probe detection logic from our previous component
func DetectOptimalBackend(libDir string) ProbeResult {
	cudaPath := filepath.Join(libDir, "libggml-cuda.so")
	if fileExists(cudaPath) && hasCudaEnvironment() {
		return ProbeResult{
			Mode: ModeCUDA, LibraryName: "libggml-cuda.so", LibraryPath: cudaPath,
			Details: "NVIDIA CUDA acceleration dynamically mapped.",
		}
	}

	cpuVariant := detectCpuVariant(libDir)
	if cpuVariant.Mode != ModeCPUFallback {
		return cpuVariant
	}

	basePath := filepath.Join(libDir, "libllama.so")
	return ProbeResult{
		Mode: ModeCPUFallback, LibraryName: "libllama.so", LibraryPath: basePath,
		Details: "Generic CPU runtime dynamic fallback.",
	}
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func hasCudaEnvironment() bool {
	if _, present := os.LookupEnv("CUDA_PATH"); present {
		return true
	}
	return fileExists("/dev/nvidia0") || fileExists("/proc/driver/nvidia/version")
}

func detectCpuVariant(libDir string) ProbeResult {
	if runtime.GOOS != "linux" {
		return ProbeResult{Mode: ModeCPUFallback}
	}
	content, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return ProbeResult{Mode: ModeCPUFallback}
	}
	cpuInfo := string(content)

	if strings.Contains(cpuInfo, "avx512f") {
		p := filepath.Join(libDir, "libggml-cpu-icelake.so")
		if fileExists(p) {
			return ProbeResult{Mode: ModeAVX512, LibraryName: "libggml-cpu-icelake.so", LibraryPath: p, Details: "AVX-512 instruction set optimized variant loaded."}
		}
	}
	if strings.Contains(cpuInfo, "avx2") {
		p := filepath.Join(libDir, "libggml-cpu-haswell.so")
		if fileExists(p) {
			return ProbeResult{Mode: ModeAVX2, LibraryName: "libggml-cpu-haswell.so", LibraryPath: p, Details: "AVX2 instruction set optimized variant loaded."}
		}
	}
	return ProbeResult{Mode: ModeCPUFallback}
}
