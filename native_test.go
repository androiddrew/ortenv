package ortenv

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sync"
	"testing"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

func nativeLibrary(t *testing.T, variable string) string {
	t.Helper()
	path := os.Getenv(variable)
	if path == "" {
		t.Skipf("set %s to opt in to native runtime checks", variable)
	}
	return path
}

// Each native scenario gets a fresh process. Resetting the coordinator or
// destroying the runtime between tests would defeat process-lifetime ownership.
// CUDA lease cycles must all execute within the same child process.
func nativeSubprocess(t *testing.T) bool {
	t.Helper()
	const marker = "ORTENV_TEST_SUBPROCESS"
	if os.Getenv(marker) == t.Name() {
		return true
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, executable,
		"-test.run=^"+regexp.QuoteMeta(t.Name())+"$", "-test.count=1", "-test.timeout=45s", "-test.v")
	cmd.Env = append(os.Environ(), marker+"="+t.Name())
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("native subprocess failed: %v\n%s", err, output)
	}
	t.Logf("native subprocess:\n%s", output)
	return false
}

func TestNativeSharedLeases(t *testing.T) {
	path := nativeLibrary(t, "ORTENV_TEST_LIBRARY")
	if !nativeSubprocess(t) {
		return
	}
	var leases []*Lease
	t.Cleanup(func() {
		for _, lease := range leases {
			if err := lease.Close(); err != nil {
				t.Error(err)
			}
		}
	})
	for range 5 {
		lease, err := Acquire(path)
		if err != nil {
			t.Fatal(err)
		}
		leases = append(leases, lease)
	}
	expectAcquireError(t, path+".different", "different library selector")
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			lease, err := Acquire(path)
			if err != nil {
				t.Error(err)
				return
			}
			if err := lease.Close(); err != nil {
				t.Error(err)
			}
			if err := lease.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	wg.Wait()
	for _, lease := range leases[:4] {
		if err := lease.Close(); err != nil {
			t.Fatal(err)
		}
		if !ort.IsInitialized() {
			t.Fatal("released another component's environment")
		}
	}
	if err := leases[4].Close(); err != nil {
		t.Fatal(err)
	}
	if !ort.IsInitialized() {
		t.Fatal("last Close destroyed the environment")
	}
	expectAcquireError(t, path+".different", "different library selector")
	lease, err := Acquire(path)
	if err != nil {
		t.Fatal(err)
	}
	leases = append(leases, lease)
	options, err := ort.NewSessionOptions()
	if err != nil {
		t.Fatal(err)
	}
	if err := options.Destroy(); err != nil {
		t.Fatal(err)
	}
	if err := lease.Close(); err != nil {
		t.Fatal(err)
	}
	if state.users != 0 || !ort.IsInitialized() {
		t.Fatal("reacquisition lost process ownership")
	}
}

func TestNativeForeignEnvironment(t *testing.T) {
	path := nativeLibrary(t, "ORTENV_TEST_LIBRARY")
	if !nativeSubprocess(t) {
		return
	}
	ort.SetSharedLibraryPath(path)
	if err := ort.InitializeEnvironment(); err != nil {
		t.Fatal(err)
	}
	expectAcquireError(t, path, "outside ortenv")
	if !ort.IsInitialized() {
		t.Fatal("destroyed foreign environment")
	}
}

func TestNativeInitializationFailureAllowsRetry(t *testing.T) {
	path := nativeLibrary(t, "ORTENV_TEST_LIBRARY")
	if !nativeSubprocess(t) {
		return
	}
	missing := filepath.Join(t.TempDir(), "missing-runtime.so")
	expectAcquireError(t, missing, "initialize ONNX Runtime")
	if ort.IsInitialized() || state.users != 0 || state.library != "" {
		t.Fatal("failed initialization took ownership")
	}
	lease := acquireLease(t, path)
	if err := lease.Close(); err != nil {
		t.Fatal(err)
	}
	if state.users != 0 || !ort.IsInitialized() {
		t.Fatal("successful retry did not retain the environment")
	}
}
