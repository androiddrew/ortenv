package ortenv

import (
	"bytes"
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
	// A child that matched no test or skipped also exits 0.
	if !bytes.Contains(output, []byte("--- PASS: "+t.Name()+" (")) {
		t.Fatalf("native subprocess did not pass %s:\n%s", t.Name(), output)
	}
	t.Logf("native subprocess:\n%s", output)
	return false
}

func TestNativeSharedInit(t *testing.T) {
	path := nativeLibrary(t, "ORTENV_TEST_LIBRARY")
	if !nativeSubprocess(t) {
		return
	}
	for range 5 {
		mustInit(t, path)
	}
	expectInitError(t, path+".different", "different library selector")
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			if err := Init(path); err != nil {
				t.Error(err)
			}
		})
	}
	wg.Wait()
	// Create and destroy native objects across several components' Init calls.
	for range 3 {
		mustInit(t, path)
		options, err := ort.NewSessionOptions()
		if err != nil {
			t.Fatal(err)
		}
		if err := options.Destroy(); err != nil {
			t.Fatal(err)
		}
	}
	if !ort.IsInitialized() || state.library == "" {
		t.Fatal("repeated Init lost process ownership")
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
	expectInitError(t, path, "outside ortenv")
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
	expectInitError(t, missing, "initialize ONNX Runtime")
	if ort.IsInitialized() || state.library != "" {
		t.Fatal("failed initialization took ownership")
	}
	mustInit(t, path)
	if !ort.IsInitialized() {
		t.Fatal("successful retry did not initialize the environment")
	}
}
