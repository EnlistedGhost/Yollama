package ml

import (
	"os"
	"path/filepath"
	"runtime"
)

type libYollamaPathSearch struct {
	executable string
	workingDir string
	goos       string
	goarch     string
}

// LibYollamaPath is the root used to find bundled llama.cpp and MLX runtime
// libraries. GPU-specific libraries live in backend subdirectories such as
// cuda_v12, rocm_v7_2, vulkan, and mlx_cuda_v13.
var LibYollamaPath = func() string {
	exe, err := os.Executable()
	if err != nil {
		return ""
	}
	if eval, err := filepath.EvalSymlinks(exe); err == nil {
		exe = eval
	}

	cwd, err := os.Getwd()
	if err != nil {
		cwd = ""
	}

	return findLibYollamaPath(libYollamaPathSearch{
		executable: exe,
		workingDir: cwd,
		goos:       runtime.GOOS,
		goarch:     runtime.GOARCH,
	})
}()

func findLibYollamaPath(search libYollamaPathSearch) string {
	candidates := libYollamaPathCandidates(search)
	for _, path := range candidates {
		if libYollamaPathExists(path) {
			return path
		}
	}

	if search.executable != "" {
		return filepath.Dir(search.executable)
	}
	return ""
}

func libYollamaPathCandidates(search libYollamaPathSearch) []string {
	goos := search.goos
	if goos == "" {
		goos = runtime.GOOS
	}
	goarch := search.goarch
	if goarch == "" {
		goarch = runtime.GOARCH
	}

	seen := map[string]bool{}
	var candidates []string
	add := func(path string) {
		if path == "" {
			return
		}
		path = filepath.Clean(path)
		if !seen[path] {
			seen[path] = true
			candidates = append(candidates, path)
		}
	}

	if search.executable != "" {
		exeDir := filepath.Dir(search.executable)
		switch goos {
		case "darwin":
			// Local dist output and standard installs keep helpers under lib/yollama.
			add(filepath.Join(exeDir, "lib", "yollama"))
			add(filepath.Join(exeDir, "..", "lib", "yollama"))
		case "linux":
			add(filepath.Join(exeDir, "..", "lib", "yollama"))
			add(filepath.Join(exeDir, "lib", "yollama"))
		case "windows":
			add(filepath.Join(exeDir, "lib", "yollama"))
			add(filepath.Join(exeDir, "..", "lib", "yollama"))
		default:
			add(filepath.Join(exeDir, "lib", "yollama"))
			add(filepath.Join(exeDir, "..", "lib", "yollama"))
		}
		addLocalLibYollamaPaths(add, exeDir, goos, goarch)
		if goos == "darwin" {
			// macOS release artifacts colocate native helpers with yollama.
			add(exeDir)
		}
	}
	addLocalLibYollamaPaths(add, search.workingDir, goos, goarch)

	return candidates
}

func addLocalLibYollamaPaths(add func(string), base, goos, goarch string) {
	if base == "" {
		return
	}
	add(filepath.Join(base, "build", "lib", "yollama"))
	add(filepath.Join(base, "dist", goos+"-"+goarch, "lib", "yollama"))
	if goos+"_"+goarch != goos+"-"+goarch {
		add(filepath.Join(base, "dist", goos+"_"+goarch, "lib", "yollama"))
	}
	if goos == "darwin" {
		add(filepath.Join(base, "dist", "darwin"))
	}
}

func libYollamaPathExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}
