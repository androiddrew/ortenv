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
	if state.users != 0 || state.library != "" {
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
		if state.users != 0 {
			t.Errorf("leaked %d leases", state.users)
		}
		state.users = 0
		state.library = ""
		binding = saved
	})
	return fake
}

func acquireLease(t *testing.T, path string) *Lease {
	t.Helper()
	lease, err := Acquire(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := lease.Close(); err != nil {
			t.Error(err)
		}
	})
	return lease
}

func TestRetainsEnvironment(t *testing.T) {
	fake := useFakeRuntime(t)
	const path = "ortenv-test-runtime.so"
	for cycle := 1; cycle <= 3; cycle++ {
		lease := acquireLease(t, path)
		if err := lease.Close(); err != nil {
			t.Fatal(err)
		}
		if !fake.initialized {
			t.Fatalf("cycle %d: last Close destroyed the environment", cycle)
		}
		if state.users != 0 || state.library != path {
			t.Fatalf("cycle %d: final Close lost ownership: users=%d library=%q", cycle, state.users, state.library)
		}
	}
	if fake.initializations != 1 || len(fake.libraries) != 1 {
		t.Fatalf("reloaded runtime: initializations=%d library selections=%v", fake.initializations, fake.libraries)
	}
}

func expectAcquireError(t *testing.T, path, message string) error {
	t.Helper()
	lease, err := Acquire(path)
	if lease != nil {
		if closeErr := lease.Close(); closeErr != nil {
			t.Error(closeErr)
		}
		t.Fatalf("failed acquisition returned a lease: %v", err)
	}
	if err == nil || !strings.Contains(err.Error(), message) {
		t.Fatalf("expected acquisition error containing %q, got %v", message, err)
	}
	return err
}

func TestNilAndZeroLeases(t *testing.T) {
	useFakeRuntime(t)
	var zero Lease
	var nilLease *Lease
	closeEmpty := func() {
		t.Helper()
		for _, lease := range []*Lease{nilLease, &zero} {
			if err := lease.Close(); err != nil {
				t.Fatal(err)
			}
		}
	}
	closeEmpty()
	if state.users != 0 {
		t.Fatal("empty lease changed ownership")
	}
	acquireLease(t, "ortenv-test-runtime.so")
	closeEmpty()
	if state.users != 1 {
		t.Fatal("empty lease released an active owner's lease")
	}
}

func TestLibrarySelectorPinned(t *testing.T) {
	fake := useFakeRuntime(t)
	const path = "ortenv-test-runtime.so"
	lease := acquireLease(t, path)
	expectAcquireError(t, path+".different", "different library selector")
	if state.users != 1 {
		t.Fatal("conflicting acquisition changed active leases")
	}
	if err := lease.Close(); err != nil {
		t.Fatal(err)
	}
	expectAcquireError(t, path+".different", "different library selector")
	if state.users != 0 {
		t.Fatal("conflicting acquisition created a lease")
	}
	acquireLease(t, path)
	if fake.initializations != 1 || len(fake.libraries) != 1 {
		t.Fatal("conflicting acquisition reconfigured the runtime")
	}
}

func TestConcurrentAcquireRelease(t *testing.T) {
	fake := useFakeRuntime(t)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for range 16 {
		wg.Go(func() {
			<-start
			for range 8 {
				lease, err := Acquire("ortenv-test-runtime.so")
				if err != nil {
					t.Error(err)
					return
				}
				for range 2 {
					if err := lease.Close(); err != nil {
						t.Error(err)
					}
				}
			}
		})
	}
	close(start)
	wg.Wait()
	if state.users != 0 || fake.initializations != 1 || !fake.initialized {
		t.Fatalf("concurrent leases lost ownership: users=%d initializations=%d initialized=%t",
			state.users, fake.initializations, fake.initialized)
	}
}

func TestConcurrentClose(t *testing.T) {
	fake := useFakeRuntime(t)
	var leases []*Lease
	for range 5 {
		leases = append(leases, acquireLease(t, "ortenv-test-runtime.so"))
	}
	if state.users != len(leases) {
		t.Fatalf("expected %d active leases, got %d", len(leases), state.users)
	}
	var wg sync.WaitGroup
	for _, lease := range leases[:4] {
		for range 8 {
			wg.Go(func() {
				if err := lease.Close(); err != nil {
					t.Error(err)
				}
			})
		}
	}
	wg.Wait()
	if state.users != 1 || !fake.initialized {
		t.Fatal("concurrent Close released another owner's lease")
	}
	for range 8 {
		wg.Go(func() {
			if err := leases[4].Close(); err != nil {
				t.Error(err)
			}
		})
	}
	wg.Wait()
	if state.users != 0 || !fake.initialized {
		t.Fatal("concurrent final Close lost process ownership")
	}
}

func TestInitializationFailureAllowsRetry(t *testing.T) {
	fake := useFakeRuntime(t)
	initErr := errors.New("initialization failed")
	fake.initializeError = initErr
	err := expectAcquireError(t, "ortenv-failed-runtime.so", "ortenv-failed-runtime.so")
	if !errors.Is(err, initErr) {
		t.Fatalf("lost initialization error: %v", err)
	}
	if state.users != 0 || state.library != "" {
		t.Fatal("failed initialization pinned the coordinator")
	}
	fake.initializeError = nil
	const path = "ortenv-test-runtime.so"
	acquireLease(t, path)
	if fake.initializations != 2 || len(fake.libraries) != 2 || fake.libraries[1] != path {
		t.Fatalf("did not retry with a new selector: initializations=%d libraries=%v", fake.initializations, fake.libraries)
	}
	if state.users != 1 || state.library != path || !fake.initialized {
		t.Fatal("successful retry did not take ownership")
	}
}

func TestForeignEnvironment(t *testing.T) {
	fake := useFakeRuntime(t)
	fake.initialized = true
	expectAcquireError(t, "ortenv-test-runtime.so", "outside ortenv")
	if fake.initializations != 0 || len(fake.libraries) != 0 || !fake.initialized {
		t.Fatal("modified a foreign environment")
	}
	if state.users != 0 || state.library != "" {
		t.Fatal("took ownership of a foreign environment")
	}
}

func TestExternalDestruction(t *testing.T) {
	for _, phase := range []string{"active", "idle"} {
		t.Run(phase, func(t *testing.T) {
			fake := useFakeRuntime(t)
			const path = "ortenv-test-runtime.so"
			lease := acquireLease(t, path)
			if phase == "idle" {
				if err := lease.Close(); err != nil {
					t.Fatal(err)
				}
			}
			fake.initialized = false
			for range 2 {
				expectAcquireError(t, path, "destroyed outside ortenv")
			}
			if fake.initializations != 1 || len(fake.libraries) != 1 || fake.initialized {
				t.Fatal("reinitialized an externally destroyed environment")
			}
			if state.library != path {
				t.Fatal("forgot the pinned selector after external destruction")
			}
		})
	}
}
