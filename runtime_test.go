package ortenv

import (
	"errors"
	"strings"
	"sync"
	"testing"
)

type fakeRuntime struct {
	initialized     bool
	initializations int
	initializeError error
	libraries       []string
}

// These tests replace process-global binding calls and must not run in parallel.
func useFakeRuntime(t *testing.T) *fakeRuntime {
	t.Helper()
	if state.library != "" {
		t.Fatal("expected a fresh coordinator")
	}
	saved := binding
	fake := &fakeRuntime{}
	binding.isInitialized = func() bool { return fake.initialized }
	binding.setLibrary = func(path string) { fake.libraries = append(fake.libraries, path) }
	binding.initialize = func() error {
		fake.initializations++
		if fake.initializeError != nil {
			return fake.initializeError
		}
		fake.initialized = true
		return nil
	}
	t.Cleanup(func() {
		state.library = ""
		binding = saved
	})
	return fake
}

func mustInit(t *testing.T, path string) {
	t.Helper()
	if err := Init(path); err != nil {
		t.Fatal(err)
	}
}

func expectInitError(t *testing.T, path, message string) error {
	t.Helper()
	err := Init(path)
	if err == nil || !strings.Contains(err.Error(), message) {
		t.Fatalf("expected Init error containing %q, got %v", message, err)
	}
	return err
}

func TestInitIsIdempotent(t *testing.T) {
	fake := useFakeRuntime(t)
	const path = "ortenv-test-runtime.so"
	for range 3 {
		mustInit(t, path)
		if !fake.initialized || state.library != path {
			t.Fatalf("Init did not take ownership: initialized=%t library=%q", fake.initialized, state.library)
		}
	}
	if fake.initializations != 1 || len(fake.libraries) != 1 {
		t.Fatalf("reloaded runtime: initializations=%d library selections=%v", fake.initializations, fake.libraries)
	}
}

func TestLibrarySelectorPinned(t *testing.T) {
	fake := useFakeRuntime(t)
	const path = "ortenv-test-runtime.so"
	mustInit(t, path)
	for range 2 {
		expectInitError(t, path+".different", "different library selector")
	}
	mustInit(t, path)
	if fake.initializations != 1 || len(fake.libraries) != 1 || state.library != path {
		t.Fatal("conflicting Init reconfigured the runtime")
	}
}

func TestConcurrentInit(t *testing.T) {
	fake := useFakeRuntime(t)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for range 16 {
		wg.Go(func() {
			<-start
			for range 8 {
				if err := Init("ortenv-test-runtime.so"); err != nil {
					t.Error(err)
					return
				}
			}
		})
	}
	close(start)
	wg.Wait()
	if fake.initializations != 1 || !fake.initialized {
		t.Fatalf("concurrent Init lost ownership: initializations=%d initialized=%t",
			fake.initializations, fake.initialized)
	}
}

func TestInitializationFailureAllowsRetry(t *testing.T) {
	fake := useFakeRuntime(t)
	initErr := errors.New("initialization failed")
	fake.initializeError = initErr
	err := expectInitError(t, "ortenv-failed-runtime.so", "ortenv-failed-runtime.so")
	if !errors.Is(err, initErr) {
		t.Fatalf("lost initialization error: %v", err)
	}
	if state.library != "" {
		t.Fatal("failed initialization pinned the coordinator")
	}
	fake.initializeError = nil
	const path = "ortenv-test-runtime.so"
	mustInit(t, path)
	if fake.initializations != 2 || len(fake.libraries) != 2 || fake.libraries[1] != path {
		t.Fatalf("did not retry with a new selector: initializations=%d libraries=%v", fake.initializations, fake.libraries)
	}
	if state.library != path || !fake.initialized {
		t.Fatal("successful retry did not take ownership")
	}
}

func TestForeignEnvironment(t *testing.T) {
	fake := useFakeRuntime(t)
	fake.initialized = true
	expectInitError(t, "ortenv-test-runtime.so", "outside ortenv")
	if fake.initializations != 0 || len(fake.libraries) != 0 || !fake.initialized {
		t.Fatal("modified a foreign environment")
	}
	if state.library != "" {
		t.Fatal("took ownership of a foreign environment")
	}
}

func TestExternalDestruction(t *testing.T) {
	fake := useFakeRuntime(t)
	const path = "ortenv-test-runtime.so"
	mustInit(t, path)
	fake.initialized = false
	for range 2 {
		expectInitError(t, path, "destroyed outside ortenv")
	}
	if fake.initializations != 1 || len(fake.libraries) != 1 || fake.initialized {
		t.Fatal("reinitialized an externally destroyed environment")
	}
	if state.library != path {
		t.Fatal("forgot the pinned selector after external destruction")
	}
}
