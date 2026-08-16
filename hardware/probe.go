package hardware

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// AccelerationType defines the active backend runtime mode
type AccelerationType string

const (
	ModeCUDA AccelerationType = "cuda"
	ModeAVX512 AccelerationType = "avx512"
	ModeAVX2 AccelerationType = "avx2"
	ModeCPUFallback AccelerationType = "cpu-generic"
)

// ProbeResult holds the detection outcome and targeted library path
type ProbeResult struct {
	Mode        AccelerationType
	LibraryName string
	LibraryPath string
	Details     string
}

// DetectOptimalBackend inspects the host environment for the best available runner
func DetectOptimalBackend(libDir string) ProbeResult {
	// 1. Check for CUDA support (e.g. if libggml-cuda.so and NVIDIA driver indicators exist)
	cudaPath := filepath.Join(libDir, "libggml-cuda.so")
	if fileExists(cudaPath) && hasCudaEnvironment() {
		return ProbeResult{
			Mode:        ModeCUDA,
			LibraryName: "libggml-cuda.so",
			LibraryPath: cudaPath,
			Details:     "NVIDIA CUDA acceleration detected and verified.",
		}
	}

	// 2. Check for microarchitecture-specific CPU variants based on host flags
	cpuVariant := detectCpuVariant(libDir)
	if cpuVariant.Mode != ModeCPUFallback {
		return cpuVariant
	}

	// 3. Fallback to base library
	basePath := filepath.Join(libDir, "libllama.so")
	return ProbeResult{
		Mode:        ModeCPUFallback,
		LibraryName: "libllama.so",
		LibraryPath: basePath,
		Details:     "Using standard generic CPU runtime fallback.",
	}
}

// PrintDoctorDiagnostics outputs a robust diagnostic report (yollama --doctor)
func PrintDoctorDiagnostics(libDir string) {
	fmt.Println("==================================================")
	fmt.Println("         Yollama System & Hardware Doctor         ")
	fmt.Println("==================================================")
	fmt.Printf("OS/Arch:      %s/%s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Printf("Target Libs:  %s\n", libDir)

	result := DetectOptimalBackend(libDir)
	fmt.Printf("Selected Mode:   [%s]\n", strings.ToUpper(string(result.Mode)))
	fmt.Printf("Primary Asset:   %s\n", result.LibraryName)
	fmt.Printf("Resolved Path:   %s\n", result.LibraryPath)
	fmt.Printf("Status Note:     %s\n", result.Details)
	fmt.Println("==================================================")
}

// Helper: check if file exists on disk
func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

// Helper: basic check for CUDA execution indicators
func hasCudaEnvironment() bool {
	// Checks if NVIDIA container toolkit or standard driver paths/env vars are present
	if _, present := os.LookupEnv("CUDA_PATH"); present {
		return true
	}
	if fileExists("/dev/nvidia0") || fileExists("/proc/driver/nvidia/version") {
		return true
	}
	return false
}

// Helper: parse /proc/cpuinfo on Linux for flags like avx2, avx512f
func detectCpuVariant(libDir string) ProbeResult {
	if runtime.GOOS != "linux" {
		return ProbeResult{Mode: ModeCPUFallback}
	}

	content, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return ProbeResult{Mode: ModeCPUFallback}
	}

	cpuInfo := string(content)

	// Check for high-performance instruction sets mapping to your .so file inventory
	if strings.Contains(cpuInfo, "avx512f") {
		p := filepath.Join(libDir, "libggml-cpu-icelake.so") // or skylakex/sapphirerapids
		if fileExists(p) {
			return ProbeResult{Mode: ModeAVX512, LibraryName: "libggml-cpu-icelake.so", LibraryPath: p, Details: "AVX-512 instruction set found."}
		}
	}

	if strings.Contains(cpuInfo, "avx2") {
		p := filepath.Join(libDir, "libggml-cpu-haswell.so") // or zen4/x64 general
		if fileExists(p) {
			return ProbeResult{Mode: ModeAVX2, LibraryName: "libggml-cpu-haswell.so", LibraryPath: p, Details: "AVX2 instruction set found."}
		}
	}

	return ProbeResult{Mode: ModeCPUFallback}
}
