// Package ortenv coordinates the process-global ONNX Runtime environment across
// independent components. Every component calls Init before creating native
// sessions, options or tensors. Once initialized, the environment and its
// native library are retained until process exit.
package ortenv

import (
	"errors"
	"fmt"
	"path/filepath"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var state struct {
	sync.Mutex
	library string
}

// Keep binding calls replaceable so lifecycle tests need no native runtime.
var binding = struct {
	isInitialized func() bool
	setLibrary    func(string)
	initialize    func() error
}{
	isInitialized: ort.IsInitialized,
	setLibrary:    ort.SetSharedLibraryPath,
	initialize:    func() error { return ort.InitializeEnvironment() },
}

// Init initializes the process-global ONNX Runtime environment from library on
// the first successful call. library is a file path or an OS-loader name such
// as "libonnxruntime.so"; file paths are normalized. The first successful call
// pins the selector until process exit, and later calls with the same selector
// return nil without reloading anything. A different selector, an environment
// initialized outside ortenv, or one destroyed outside ortenv is an error.
// Compatibility is determined by the binding's C API request, not an exact
// native version string. Init is safe to call concurrently.
//
// The environment and its native library are never destroyed: unloading and
// reloading the runtime can leave CUDA providers with stale pointers. Sessions,
// options and tensors must still be destroyed normally.
func Init(library string) error {
	state.Lock()
	defer state.Unlock()
	path, err := librarySelector(library)
	if err != nil {
		return err
	}
	if state.library != "" {
		if state.library != path {
			return errors.New("ortenv: the process is pinned to a different library selector; use the same path or loader name for all components")
		}
		if !binding.isInitialized() {
			return errors.New("ortenv: ONNX Runtime environment was destroyed outside ortenv; restart the process rather than reinitializing")
		}
		return nil
	}
	if binding.isInitialized() {
		return errors.New("ortenv: ONNX Runtime environment is owned by another component outside ortenv")
	}
	binding.setLibrary(path)
	if err = binding.initialize(); err != nil {
		return fmt.Errorf("ortenv: initialize ONNX Runtime from %q (must provide the binding's requested C API): %w", path, err)
	}
	state.library = path
	return nil
}

func librarySelector(library string) (string, error) {
	if library == "" {
		return "", errors.New("ortenv: library selector is required; provide a shared-library path or OS-loader name")
	}
	// A bare name belongs to the platform loader's search path/cache. Turning it
	// into a path under the current directory would break system installations.
	if filepath.Base(library) == library && filepath.VolumeName(library) == "" {
		return library, nil
	}
	path, err := filepath.Abs(library)
	if err != nil {
		return "", err
	}
	if resolved, err := filepath.EvalSymlinks(path); err == nil {
		path = resolved
	}
	return path, nil
}
