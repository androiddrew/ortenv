package ortenv

import (
	"os"
	"strings"
	"sync"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

func TestZeroLease(t *testing.T) {
	var zero Lease
	if err := zero.Close(); err != nil {
		t.Fatal(err)
	}
	if state.users != 0 {
		t.Fatal("zero lease changed ownership")
	}
}

func runtimeLibrary(t *testing.T) string {
	t.Helper()
	path := os.Getenv("ORTENV_TEST_LIBRARY")
	if path == "" {
		t.Skip("set ORTENV_TEST_LIBRARY for native runtime ownership checks")
	}
	return path
}

func TestSharedLeases(t *testing.T) {
	path := runtimeLibrary(t)
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
	if _, err := Acquire(path + ".different"); err == nil {
		t.Fatal("accepted conflicting runtime path")
	}
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
	if ort.IsInitialized() {
		t.Fatal("last owner leaked environment")
	}
	lease, err := Acquire(path)
	if err != nil {
		t.Fatal(err)
	}
	leases = append(leases, lease)
	if err := lease.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestForeignEnvironment(t *testing.T) {
	path := runtimeLibrary(t)
	ort.SetSharedLibraryPath(path)
	if err := ort.InitializeEnvironment(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := ort.DestroyEnvironment(); err != nil {
			t.Error(err)
		}
	})
	if _, err := Acquire(path); err == nil || !strings.Contains(err.Error(), "outside ortenv") {
		t.Fatalf("expected foreign ownership error, got %v", err)
	}
	if !ort.IsInitialized() {
		t.Fatal("destroyed foreign environment")
	}
}
