# ortenv

`github.com/androiddrew/ortenv` coordinates ownership of the process-global
environment used by `github.com/yalue/onnxruntime_go`.

It is a small lifecycle manager for applications that use multiple independent
ONNX-backed components. Each component acquires a lease before creating native
sessions/options/tensors, destroys those objects, then releases the lease. The
first lease initializes the environment; the last release destroys it.

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
  `ort.InitializeEnvironment` or `ort.DestroyEnvironment` around leased sessions.
- Leases must not be copied. `Close` is idempotent, nil-safe and concurrency-safe.
- Active owners must agree on the native-library selector. Explicit file paths
  are normalized and symlinks resolved where possible. Bare loader names are
  passed to the OS unchanged, supporting system/package-manager installations.
  Use the same selector across components: a loader name and an absolute path
  are not assumed to identify the same file. Conflicts fail while owners are live.
- Provider, device and thread choices belong to **individual sessions**. The
  manager does not configure models, schedule inference or serialize session runs.
- Requires **Go 1.25.0** or newer, with CGO. The binding version is declared in
  `go.mod`; Go's minimal version selection may choose a newer one if another
  dependency requires it. Check the resolved version with
  `go list -m github.com/yalue/onnxruntime_go`.

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

Tested on Linux/amd64 with Go 1.25.0 and native ONNX Runtime 1.22.0 and 1.23.2,
using both explicit paths and loader names. To check your own installation:

```bash
ORTENV_TEST_LIBRARY=<path-or-name> GOWORK=off go test -race ./...
```

## Development

```bash
go test ./...
go vet ./...
ORTENV_TEST_LIBRARY=/path/to/libonnxruntime.so go test -race ./...
# Or use a name resolved through the OS loader's configured search path:
ORTENV_TEST_LIBRARY=libonnxruntime.so go test -race ./...
```

Native tests exercise five live owners, concurrent leases, repeat initialization,
conflicting paths and foreign-owner rejection. Without `ORTENV_TEST_LIBRARY`,
native tests explicitly skip. The concurrent tests use `WaitGroup.Go`, which sets
the Go 1.25 minimum.

## License

[MIT](LICENSE), Copyright (c) 2026 Drew Bednar. The `onnxruntime_go` binding is
[MIT licensed](https://github.com/yalue/onnxruntime_go/blob/master/LICENSE).
