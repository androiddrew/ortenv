// Package ortenv coordinates the process-global ONNX Runtime environment across
// independent components. All native users must hold a lease until
// their sessions, options and tensors have been destroyed.
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

// Acquire accepts a library file path or an OS-loader name such as
// "libonnxruntime.so". All active leases must use the same selector (file paths
// are normalized). Compatibility is determined by the binding's C API request,
// not an exact native version string. Release the lease after native resources.
func Acquire(library string) (*Lease, error) {
	state.Lock()
	defer state.Unlock()
	path, err := librarySelector(library)
	if err != nil {
		return nil, err
	}
	if state.users > 0 {
		if state.library != path {
			return nil, errors.New("ortenv: another active lease uses a different library selector; use the same path or loader name for all active leases")
		}
	} else {
		if ort.IsInitialized() {
			return nil, errors.New("ortenv: ONNX Runtime environment is owned by another component outside ortenv")
		}
		ort.SetSharedLibraryPath(path)
		if err = ort.InitializeEnvironment(); err != nil {
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

// Close releases the lease. When the last active lease is released, Close
// destroys the environment and returns any error from doing so. Close is
// idempotent, safe to call concurrently and safe on a nil Lease.
//
// If destroying the environment fails, the lease is still released and the
// coordinator no longer considers itself the owner, but the native environment
// may remain initialized. A later Acquire then fails with the error for an
// environment owned outside ortenv.
func (l *Lease) Close() error {
	state.Lock()
	defer state.Unlock()
	if l == nil || !l.active {
		return nil
	}
	l.active = false
	state.users--
	if state.users == 0 {
		state.library = ""
		return ort.DestroyEnvironment()
	}
	return nil
}
