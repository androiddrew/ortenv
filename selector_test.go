package ortenv

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoaderNamesRemainLoaderNames(t *testing.T) {
	for _, name := range []string{"libonnxruntime.so", "libonnxruntime.so.1", "onnxruntime.dll", "libonnxruntime.dylib"} {
		got, err := librarySelector(name)
		if err != nil || got != name {
			t.Fatalf("%q became %q: %v", name, got, err)
		}
	}
	if _, err := Acquire(""); err == nil {
		t.Fatal("accepted missing runtime selector")
	}
}

func TestLibraryPathsNormalizeAliases(t *testing.T) {
	path := filepath.Join(t.TempDir(), "runtime.so")
	if err := os.WriteFile(path, []byte("path fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	want, err := filepath.EvalSymlinks(path)
	if err != nil {
		t.Fatal(err)
	}
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	rel, err := filepath.Rel(cwd, path)
	if err != nil {
		t.Fatal(err)
	}
	for _, input := range []string{path, rel} {
		got, err := librarySelector(input)
		if err != nil || got != want {
			t.Fatalf("%q became %q; want %q: %v", input, got, want, err)
		}
	}
	alias := path + ".alias"
	if err := os.Symlink(path, alias); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	got, err := librarySelector(alias)
	if err != nil || got != want {
		t.Fatalf("alias became %q; want %q: %v", got, want, err)
	}
}
