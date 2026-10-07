# ortenv

`github.com/androiddrew/ortenv` coordinates ownership of the process-global
environment used by `github.com/yalue/onnxruntime_go`.

It is a small lifecycle manager for applications that use multiple independent
ONNX-backed components. Each component acquires a lease before creating native
sessions/options/tensors, destroys those objects, then releases the lease. The
first successful lease initializes the environment, which stays loaded until
process exit. Closing the last lease retains the environment for later reuse.

```go
import (
    "errors"
    "github.com/androiddrew/ortenv"
)

func run() (err error) {
    lease, err := ortenv.Acquire("libonnxruntime.so") // or an explicit file path
    if err != nil { return err }
    defer func() { err = errors.Join(err, lease.Close()) }()

    // Create sessions with onnxruntime_go here. Defer their Destroy calls AFTER
    // the lease defer so native objects are destroyed before the final release.
    return nil
}
```

## Status

This module is pre-1.0. The `Acquire` and `Lease` API may change between minor
releases until a v1 release is made.

## Ownership contract

- All components using the binding in the same process must use the coordinator.
  Direct foreign environment ownership is rejected; do not independently call
  `ort.InitializeEnvironment` or `ort.DestroyEnvironment` once ortenv manages the
  environment, including periods with no active leases. If `Acquire` detects
  external destruction, it returns an error requiring a process restart.
- Leases must not be copied. `Close` is idempotent, nil-safe and concurrency-safe.
  It always returns nil; callers still destroy their native resources first.
- The first successful acquisition pins the native-library selector for the
  lifetime of the process. Explicit file paths are normalized and symlinks
  resolved where possible. Bare loader names are passed to the OS unchanged,
  supporting system/package-manager installations. Use the same selector across
  components: a loader name and an absolute path are not assumed to identify the
  same file. Conflicts fail even after all leases close; switching runtimes
  requires a new process. A failed initialization does not pin the selector.
- Provider, device and thread choices belong to **individual sessions**. The
  manager does not configure models, schedule inference or serialize session runs.
- Requires **Go 1.25.0** or newer, with CGO. The binding version is declared in
  `go.mod`; Go's minimal version selection may choose a newer one if another
  dependency requires it. Check the resolved version with
  `go list -m github.com/yalue/onnxruntime_go`.

### Why retain the environment?

In v0.1.0, closing the final lease destroyed the environment. The binding's
`DestroyEnvironment` also unloads the native library. CUDA provider libraries
can remain loaded with pointers into that old library mapping, causing a crash
when a later acquisition reloads the runtime
([issue #2](https://github.com/androiddrew/ortenv/issues/2)).

The environment and library are now retained for both CPU and CUDA use. Their
process-level resources remain resident until exit; per-component sessions,
options and tensors must still be destroyed normally. No extra keep-alive lease
is needed to bridge gaps between components.

## Native runtime

Install a native ONNX Runtime however you choose: a system package manager, an
upstream archive, or a custom installation. A configured loader may resolve
`libonnxruntime.so`; an explicit path can select a particular install. This
module downloads and distributes no native runtime, CUDA libraries or model
assets.

Initialization asks the binding to load the C API it requires; unsupported
runtimes fail at that point. ortenv adds no version check of its own. A
successful `Acquire` does not guarantee that a particular model, operator or
execution provider is supported; validate those with your own sessions.

## Development

### GPU-free checks

With native-test opt-in variables unset, `go test ./...` uses a fake runtime for
lifecycle checks and filesystem fixtures for selector checks. It does not load
ONNX Runtime. Tests cover retained ownership across lease cycles, pinned
selectors, concurrent acquisitions and repeated closes, initialization failure
and retry, foreign ownership, and external destruction. The concurrent tests use
`WaitGroup.Go`, which sets the Go 1.25 minimum.

For explicit GPU-free checks with limited parallelism on a shared Linux host:

```bash
env -u ORTENV_TEST_LIBRARY -u ORTENV_TEST_CUDA_LIBRARY \
    GOFLAGS= GOWORK=off GOMAXPROCS=2 \
    nice -n 15 go test -race -p 1 -parallel 1 -count=1 ./...
env -u ORTENV_TEST_LIBRARY -u ORTENV_TEST_CUDA_LIBRARY \
    GOFLAGS= GOWORK=off GOMAXPROCS=2 \
    nice -n 15 go vet -p 1 ./...
```

These still consume CPU, RAM and disk I/O. Avoiding native runtime calls, rather
than just hiding devices with `CUDA_VISIBLE_DEVICES`, keeps these checks off the
GPU.

### Native lifecycle checks (opt-in)

```bash
ORTENV_TEST_LIBRARY=/path/to/libonnxruntime.so GOWORK=off go test -race -count=1 ./...
# Or use a name resolved through the OS loader's configured search path:
ORTENV_TEST_LIBRARY=libonnxruntime.so GOWORK=off go test -race -count=1 ./...
```

Native tests exercise five live owners, concurrent leases, reuse after the final
close, conflicting selectors, foreign-owner rejection, and initialization
failure followed by a successful retry. Each scenario runs in a fresh subprocess
so it does not need to reset or unload a process-lifetime runtime. Without
`ORTENV_TEST_LIBRARY`, these tests explicitly skip.

### CUDA regression (explicit opt-in)

```bash
ORTENV_TEST_CUDA_LIBRARY=/path/to/cuda/libonnxruntime.so GOWORK=off \
    go test -tags=cuda -run '^TestCUDALeaseCycles$' -count=1 -v .
```

Both the `cuda` build tag and `ORTENV_TEST_CUDA_LIBRARY` are required. Setting
`ORTENV_TEST_LIBRARY` alone does not enable the CUDA test. Once enabled, missing
provider libraries or CUDA errors fail the test rather than skip it.

The regression creates, updates and destroys CUDA provider options across three
acquire/final-close cycles within one subprocess. It does not run inference, but
updating `device_id` invokes CUDA device enumeration. **Run it only when GPU
access is acceptable.** When protecting an active training run, defer this test
and native tests against a CUDA-enabled runtime until the run finishes. A test
subprocess isolates a crash from the test runner, not access to the shared GPU.

A utility script, `./scripts/ort-manage.sh`, is provided for developers to 
manage Onnxruntime versions (CPU and Cuda) on a Linux development amd64 machine.

## License

[MIT](LICENSE), Copyright (c) 2026 Drew Bednar. The `onnxruntime_go` binding is
[MIT licensed](https://github.com/yalue/onnxruntime_go/blob/master/LICENSE).
