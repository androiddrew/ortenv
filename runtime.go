// Package ortenv coordinates the process-global ONNX Runtime environment across
// independent components. All native users must hold a lease until
// their sessions, options and tensors have been destroyed. Once initialized,
// the environment and its native library are retained until process exit.
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
	users   int
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

// Acquire accepts a library file path or an OS-loader name such as
// "libonnxruntime.so". The first successful acquisition pins the selector for
// the lifetime of the process, including periods with no active leases. All
// acquisitions must use the same selector (file paths are normalized).
// Compatibility is determined by the binding's C API request, not an exact
// native version string. Release the lease after native resources.
func Acquire(library string) (*Lease, error) {
	state.Lock()
	defer state.Unlock()
	path, err := librarySelector(library)
	if err != nil {
		return nil, err
	}
	if state.library != "" {
		if state.library != path {
			return nil, errors.New("ortenv: the process is pinned to a different library selector; use the same path or loader name for all acquisitions")
		}
		if !binding.isInitialized() {
			return nil, errors.New("ortenv: ONNX Runtime environment was destroyed outside ortenv; restart the process rather than reinitializing")
		}
	} else {
		if binding.isInitialized() {
			return nil, errors.New("ortenv: ONNX Runtime environment is owned by another component outside ortenv")
		}
		binding.setLibrary(path)
		if err = binding.initialize(); err != nil {
			return nil, fmt.Errorf("ortenv: initialize ONNX Runtime from %q (must provide the binding's requested C API): %w", path, err)
		}
		state.library = path
	}
	state.users++
	return &Lease{active: true}, nil
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

// Lease is a share of the process-global ONNX Runtime environment, obtained
// from Acquire. A Lease must not be copied.
type Lease struct{ active bool }

// Close releases the lease, retaining the environment and native library even
// after the last lease is closed. Unloading and reloading the runtime can leave
// CUDA providers with stale pointers, so the runtime stays loaded until process
// exit. Sessions, options and tensors must still be destroyed before Close.
//
// Close always returns nil. It is idempotent, safe to call concurrently and safe
// on a nil Lease.
func (l *Lease) Close() error {
	state.Lock()
	defer state.Unlock()
	if l == nil || !l.active {
		return nil
	}
	l.active = false
	state.users--
	return nil
}
