//go:build cuda

package ortenv

import (
	"fmt"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

// This test calls CUDA even though it does not create a session or run a model.
// Both the cuda build tag and ORTENV_TEST_CUDA_LIBRARY are required to opt in.
func TestCUDALeaseCycles(t *testing.T) {
	path := nativeLibrary(t, "ORTENV_TEST_CUDA_LIBRARY")
	if !nativeSubprocess(t) {
		return
	}
	for cycle := 1; cycle <= 3; cycle++ {
		if !t.Run(fmt.Sprintf("cycle_%d", cycle), func(t *testing.T) {
			acquireLease(t, path)
			cuda, err := ort.NewCUDAProviderOptions()
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := cuda.Destroy(); err != nil {
					t.Error(err)
				}
			})
			if err := cuda.Update(map[string]string{"device_id": "0"}); err != nil {
				t.Fatal(err)
			}
		}) {
			return
		}
		// Subtest cleanup destroys the options, then closes the final lease.
		if state.users != 0 || !ort.IsInitialized() {
			t.Fatalf("cycle %d: final Close did not retain the environment", cycle)
		}
	}
}
