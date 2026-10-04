#!/usr/bin/env bash
# Install, switch between, and verify side-by-side ONNX Runtime releases.
#
# Purpose
#   Programs that load ONNX Runtime at run time (for example Go code using
#   github.com/yalue/onnxruntime_go) need a native libonnxruntime.so that
#   provides the C API version their binding was built for. This script manages
#   the official prebuilt Linux releases from github.com/microsoft/onnxruntime so
#   several versions can be installed at once, one made the system default, and
#   any of them tested explicitly by path:
#
#     ort-manage.sh install 1.22.0                  # CUDA 12 build; becomes current if none is
#     ort-manage.sh install 1.30.0 --variant cpu    # installed alongside, not made current
#     ./myapp --ort-lib "$(ort-manage.sh path 1.30.0-cpu)"   # e.g. into ort.SetSharedLibraryPath
#
#   `verify` checks that an install is usable: the core library and CUDA provider
#   resolve all their shared-library dependencies, the library loads and serves
#   the requested C API version, and (for the current install) the dynamic linker
#   finds it by name. For an optional provider with missing dependencies, such as
#   TensorRT, it prints the matching packages to install.
#
# Layout
#   /opt/onnxruntime/<version>-<variant>/{include,lib}   one dir per install
#   /opt/onnxruntime/current -> <version>-<variant>      active install
#   /etc/ld.so.conf.d/onnxruntime.conf                   points at current/lib
#   ~/.cache/onnxruntime/                                downloaded tarballs, CI configs
#
#   Variants: cpu, cuda12, cuda13. GPU builds also contain the CPU provider, so a
#   GPU install serves both; cpu is the smaller CPU-only package.
#
# Requirements
#   bash 4+, curl, jq, sha256sum, tar, python3 (load test), ldd, and sudo.
#
# Limitations
#   - Linux only: x86_64 (cpu, cuda12, cuda13) and aarch64 (cpu). No macOS,
#     Windows, CUDA 11 builds, or other execution providers (ROCm, OpenVINO, ...).
#   - Only official release tarballs. Custom or source builds can be placed under
#     ORT_ROOT by hand and used with `use`/`verify`, but not installed by this.
#   - Checksums are only verified when GitHub publishes a digest for the asset
#     (recent releases). For older ones the hash is recorded in INSTALL_INFO but
#     not checked against anything.
#   - `current` and the ld.so.conf entry are system-wide: switching affects every
#     user and process that loads libonnxruntime.so by name. Changing ORT_ROOT
#     repoints that single entry rather than adding a second one.
#   - It does not install CUDA, cuDNN, TensorRT or GPU drivers. `verify` only
#     shows that libraries link and the C API loads; it does not check driver
#     compatibility or run a model on the GPU. Dependency checks use ldd in the
#     caller's environment, so LD_LIBRARY_PATH can change the result.
#   - TensorRT hints read the version from the release's CI config in the
#     onnxruntime repo. Releases before 1.21 don't record it there, upstream may
#     move the file, and the suggested commands are apt (Debian/Ubuntu) only.
#     apt allows one TensorRT version system-wide.
#   - Asset names are matched against upstream's naming schemes as of 1.30; a new
#     scheme needs a pick_asset update.
#   - Unauthenticated GitHub API calls are limited to 60/hour; set GITHUB_TOKEN to
#     raise that.
#   - DEFAULT_API matches onnxruntime_go v1.22; pass `verify --api N` or change
#     it for other bindings.
#   - Not safe to run concurrently, and `remove` leaves cached downloads in place.

set -euo pipefail

ORT_ROOT="${ORT_ROOT:-/opt/onnxruntime}"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/onnxruntime"
LD_CONF=/etc/ld.so.conf.d/onnxruntime.conf
REPO=microsoft/onnxruntime
DEFAULT_VARIANT=cuda12
DEFAULT_API=22 # C API version required by github.com/yalue/onnxruntime_go v1.22

die() { echo "error: $*" >&2; exit 1; }
info() { echo "==> $*"; }
warn() { echo "warning: $*" >&2; }

usage() {
	cat <<EOF
Usage: $(basename "$0") <command> [args]

Commands:
  install <version> [--variant cpu|cuda12|cuda13] [--use]
                       Download, checksum, and install a release to
                       $ORT_ROOT/<version>-<variant>, then verify it.
                       Becomes current if --use is given or nothing is current.
  use <name>           Make <name> (e.g. 1.22.0-cuda12) the current install.
  list                 List installed runtimes; * marks current.
  available [n]        List the n most recent upstream releases (default 10).
  verify [name] [--api N]
                       Check files, linkage, and load the library to confirm it
                       serves C API N (default $DEFAULT_API). Defaults to current.
  path [name]          Print the libonnxruntime.so path, to hand to a program
                       that takes a library path, e.g.
                       ./myapp --ort-lib "\$($(basename "$0") path 1.30.0-cpu)".
  remove <name>        Delete an install (refuses to remove current).

Environment: ORT_ROOT (default /opt/onnxruntime), GITHUB_TOKEN (optional,
avoids API rate limits).
EOF
}

gh_api() {
	local auth=()
	[[ -n "${GITHUB_TOKEN:-}" ]] && auth=(-H "Authorization: Bearer $GITHUB_TOKEN")
	curl -fsSL "${auth[@]}" -H "Accept: application/vnd.github+json" "https://api.github.com/repos/$REPO/$1"
}

version_ge() { [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" == "$2" ]]; }

# Resolve the install name from an argument, defaulting to current.
resolve_name() {
	local name="${1:-}"
	if [[ -z "$name" ]]; then
		[[ -L "$ORT_ROOT/current" ]] || die "no current install; pass a name"
		name="$(basename "$(readlink "$ORT_ROOT/current")")"
	fi
	[[ -d "$ORT_ROOT/$name/lib" ]] || die "not installed: $name (see '$(basename "$0") list')"
	echo "$name"
}

# Pick the release asset for a version/variant. Upstream naming has changed over
# time: gpu-<v> (CUDA 12 from 1.19), gpu-cuda12-<v>, gpu_cuda12-<v>, gpu_cuda13-<v>.
pick_asset() {
	local version="$1" variant="$2" assets="$3" arch candidates=()
	arch="$(uname -m)"
	case "$arch:$variant" in
	x86_64:cpu) candidates=("onnxruntime-linux-x64-$version.tgz") ;;
	x86_64:cuda12)
		candidates=("onnxruntime-linux-x64-gpu_cuda12-$version.tgz" "onnxruntime-linux-x64-gpu-cuda12-$version.tgz")
		# Before 1.19 the plain gpu tarball was built against CUDA 11.
		version_ge "$version" 1.19.0 && candidates+=("onnxruntime-linux-x64-gpu-$version.tgz")
		;;
	x86_64:cuda13) candidates=("onnxruntime-linux-x64-gpu_cuda13-$version.tgz") ;;
	aarch64:cpu) candidates=("onnxruntime-linux-aarch64-$version.tgz") ;;
	*) die "unsupported arch/variant: $arch/$variant" ;;
	esac
	local c
	for c in "${candidates[@]}"; do
		if grep -qxF "$c" <<<"$assets"; then
			echo "$c"
			return
		fi
	done
	die "no $variant asset for $version on $arch. Linux assets in this release:
$(grep linux <<<"$assets" || echo '  (none)')"
}

ensure_ld_conf() {
	if [[ "$(cat "$LD_CONF" 2>/dev/null)" != "$ORT_ROOT/current/lib" ]]; then
		info "Registering $ORT_ROOT/current/lib in $LD_CONF"
		echo "$ORT_ROOT/current/lib" | sudo tee "$LD_CONF" >/dev/null
	fi
	sudo ldconfig
}

# Print the TensorRT .deb version (e.g. 10.9.0.34-1+cuda12.8) a release was built
# against, read from that tag's CI config. Prints nothing if it isn't recorded
# there (releases before 1.21) or the config can't be fetched.
trt_deb_version() {
	local version="$1" cuda="$2" cached="$CACHE_DIR/ci-variables-$1.yml" value ref
	if [[ ! -s "$cached" ]]; then
		mkdir -p "$CACHE_DIR"
		curl -fsSL -o "$cached" \
			"https://raw.githubusercontent.com/$REPO/v$version/tools/ci_build/github/azure-pipelines/templates/common-variables.yml" \
			2>/dev/null || { rm -f "$cached"; return 0; }
	fi
	value="$(sed -n "s/^ *linux_trt_version_cuda$cuda: *//p" "$cached" | tr -d "'\"")"
	[[ -n "$value" ]] || return 0
	# Values may reference another variable: ${{ variables.common_trt_version }}-1.cuda12.8
	if [[ "$value" =~ \$\{\{\ *variables\.([A-Za-z0-9_]+)\ *\}\} ]]; then
		ref="$(sed -n "s/^ *${BASH_REMATCH[1]}: *//p" "$cached" | tr -d "'\"")"
		value="${value/"${BASH_REMATCH[0]}"/$ref}"
	fi
	# CI stores the RPM form; Debian packages use '-1+' instead of '-1.'.
	echo "${value/-1./-1+}"
}

# Explain how to satisfy the TensorRT provider's missing libraries for an install.
trt_hint() {
	local name="$1" missing="$2" version="${1%-*}" variant="${1##*-}" deb pkgs=() so
	[[ "$variant" == cuda* ]] || return 0
	# Map each missing soname (libnvinfer_plugin.so.10) to its package (libnvinfer-plugin10).
	while read -r so; do
		case "$so" in
		libnvinfer.so.*) pkgs+=("libnvinfer${so##*.so.}") ;;
		libnvinfer_plugin.so.*) pkgs+=("libnvinfer-plugin${so##*.so.}") ;;
		libnvonnxparser.so.*) pkgs+=("libnvonnxparsers${so##*.so.}") ;;
		esac
	done < <(awk '{print $1}' <<<"$missing")
	((${#pkgs[@]})) || return 0

	deb="$(trt_deb_version "$version" "${variant#cuda}")"
	echo "         TensorRT is optional. To enable it, install the build $version was made with:"
	if [[ -z "$deb" ]]; then
		echo "           (this release's CI config doesn't record it; see the requirements table at"
		echo "            https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)"
		echo "           apt-cache madison ${pkgs[0]} | grep +$variant   # list candidate versions"
		return 0
	fi
	echo "           sudo apt install$(printf ' %s='"$deb" "${pkgs[@]}")"
	echo "           sudo apt-mark hold ${pkgs[*]}   # keep apt upgrade from replacing it"
	local available # captured first: grep -q in a pipe can SIGPIPE apt-cache under pipefail
	available="$(apt-cache madison "${pkgs[0]}" 2>/dev/null || true)"
	if command -v apt-cache >/dev/null && ! grep -qF "$deb" <<<"$available"; then
		echo "         $deb is not in your apt sources; add NVIDIA's CUDA repo for your distro"
		echo "         (https://developer.nvidia.com/cuda-downloads) or use NVIDIA's TensorRT tar package."
	fi
	echo "         apt installs one TensorRT system-wide; other ORT installs may want a different one."
}

cmd_install() {
	local version="" variant="$DEFAULT_VARIANT" activate=0
	while (($#)); do
		case "$1" in
		--variant) variant="${2:?--variant needs a value}"; shift 2 ;;
		--use) activate=1; shift ;;
		-*) die "unknown flag: $1" ;;
		*) version="${1#v}"; shift ;;
		esac
	done
	[[ -n "$version" ]] || die "usage: install <version> [--variant cpu|cuda12|cuda13] [--use]"
	[[ "$(uname -m)" == aarch64 && "$variant" == "$DEFAULT_VARIANT" ]] && variant=cpu

	local name="$version-$variant" dest="$ORT_ROOT/$version-$variant"
	[[ -e "$dest" ]] && die "$name already installed at $dest (remove it first to reinstall)"

	info "Looking up release v$version"
	local release
	release="$(gh_api "releases/tags/v$version")" || die "release v$version not found"
	local asset digest
	asset="$(pick_asset "$version" "$variant" "$(jq -r '.assets[].name' <<<"$release")")"
	digest="$(jq -r --arg n "$asset" '.assets[] | select(.name == $n) | .digest // empty' <<<"$release")"

	mkdir -p "$CACHE_DIR"
	local tarball="$CACHE_DIR/$asset"
	if [[ ! -f "$tarball" ]]; then
		info "Downloading $asset"
		curl -fL --progress-bar -o "$tarball.part" \
			"https://github.com/$REPO/releases/download/v$version/$asset"
		mv "$tarball.part" "$tarball"
	else
		info "Using cached $tarball"
	fi

	local actual
	actual="sha256:$(sha256sum "$tarball" | cut -d' ' -f1)"
	if [[ -n "$digest" ]]; then
		if [[ "$actual" != "$digest" ]]; then
			rm -f "$tarball"
			die "checksum mismatch for $asset: expected $digest, got $actual (cached file removed)"
		fi
		info "Checksum OK ($digest)"
	else
		warn "upstream publishes no digest for $asset; recording $actual without verification"
	fi

	info "Installing to $dest"
	sudo mkdir -p "$ORT_ROOT"
	local staging
	staging="$(sudo mktemp -d "$ORT_ROOT/.staging.XXXXXX")"
	# Expand now: $staging is local and gone by the time an EXIT trap runs.
	# shellcheck disable=SC2064
	trap "sudo rm -rf '$staging'" EXIT
	sudo tar --no-same-owner -xzf "$tarball" -C "$staging" --strip-components=1
	# The Go binding loads the unversioned name; make sure it exists.
	# Staging is root-owned mode 700 until the chmod below, so test as root.
	if ! sudo test -e "$staging/lib/libonnxruntime.so"; then
		sudo ln -s libonnxruntime.so.1 "$staging/lib/libonnxruntime.so"
	fi
	printf 'asset=%s\n%s\n' "$asset" "$actual" | sudo tee "$staging/INSTALL_INFO" >/dev/null
	sudo chmod 755 "$staging"
	sudo mv "$staging" "$dest"
	trap - EXIT

	if ((activate)) || [[ ! -L "$ORT_ROOT/current" ]]; then
		cmd_use "$name"
	else
		info "Installed $name (current is still $(basename "$(readlink "$ORT_ROOT/current")"); run 'use $name' to switch)"
	fi
	cmd_verify "$name"
}

cmd_use() {
	local name
	name="$(resolve_name "${1:?usage: use <name>}")"
	info "Switching current -> $name"
	sudo ln -sfn "$name" "$ORT_ROOT/current"
	ensure_ld_conf
}

cmd_list() {
	local current="" d
	[[ -L "$ORT_ROOT/current" ]] && current="$(basename "$(readlink "$ORT_ROOT/current")")"
	shopt -s nullglob
	local found=0
	for d in "$ORT_ROOT"/*/; do
		d="$(basename "$d")"
		[[ "$d" == current || "$d" == .* ]] && continue
		found=1
		if [[ "$d" == "$current" ]]; then echo "* $d"; else echo "  $d"; fi
	done
	((found)) || echo "(nothing installed under $ORT_ROOT)"
}

cmd_available() {
	gh_api "releases?per_page=${1:-10}" |
		jq -r '.[] | select((.draft | not) and (.tag_name | test("^v[0-9]"))) | "\(.tag_name | ltrimstr("v"))\t\(.published_at[:10])\t\([.assets[].name | select(test("linux-x64-gpu")) | capture("gpu[-_]?(?<v>cuda1[0-9])?-").v // "cuda"] | unique | join(","))"' |
		awk -F'\t' 'BEGIN { printf "%-10s %-12s %s\n", "VERSION", "PUBLISHED", "GPU BUILDS" } { printf "%-10s %-12s %s\n", $1, $2, $3 }'
}

cmd_verify() {
	local name="" api="$DEFAULT_API"
	while (($#)); do
		case "$1" in
		--api) api="${2:?--api needs a value}"; shift 2 ;;
		*) name="$1"; shift ;;
		esac
	done
	name="$(resolve_name "$name")"
	local lib="$ORT_ROOT/$name/lib" fail=0
	info "Verifying $name"

	if [[ -e "$lib/libonnxruntime.so" ]]; then
		echo "  ok   $lib/libonnxruntime.so -> $(readlink -f "$lib/libonnxruntime.so" | xargs basename)"
	else
		echo "  FAIL missing $lib/libonnxruntime.so"; fail=1
	fi

	local so missing
	for so in "$lib"/libonnxruntime.so "$lib"/libonnxruntime_providers_*.so; do
		[[ -e "$so" ]] || continue
		missing="$(ldd "$so" 2>&1 | grep 'not found' || true)"
		if [[ -n "$missing" ]]; then
			# Only the core library and the CUDA provider matter here; others
			# (e.g. TensorRT) are optional extras that ship in GPU tarballs.
			case "$(basename "$so")" in
			libonnxruntime.so | libonnxruntime_providers_shared.so | libonnxruntime_providers_cuda.so)
				echo "  FAIL $(basename "$so") has unresolved dependencies:"
				fail=1
				;;
			*) echo "  warn $(basename "$so") (optional) has unresolved dependencies:" ;;
			esac
			sed 's/^/         /' <<<"$missing"
			[[ "$(basename "$so")" == libonnxruntime_providers_tensorrt.so ]] && trt_hint "$name" "$missing"
		else
			echo "  ok   $(basename "$so") dependencies resolve"
		fi
	done

	# Load the library and ask it for its version and the requested C API table.
	local out
	if out="$(python3 - "$lib/libonnxruntime.so" "$api" 2>&1 <<'PY'
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
class OrtApiBase(ctypes.Structure):
    _fields_ = [("GetApi", ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint32)),
                ("GetVersionString", ctypes.CFUNCTYPE(ctypes.c_char_p))]
lib.OrtGetApiBase.restype = ctypes.POINTER(OrtApiBase)
base = lib.OrtGetApiBase().contents
print(base.GetVersionString().decode())
sys.exit(0 if base.GetApi(int(sys.argv[2])) else 1)
PY
	)"; then
		echo "  ok   loads; runtime $(head -1 <<<"$out") provides C API $api"
	else
		echo "  FAIL load test: runtime does not provide C API $api"
		sed 's/^/         /' <<<"$out"
		fail=1
	fi

	if [[ "$(readlink "$ORT_ROOT/current" 2>/dev/null)" == "$name" ]]; then
		# Capture first: under pipefail, grep -q exiting early can SIGPIPE ldconfig
		# and fail the pipeline even when the entry is present.
		local cache
		cache="$(ldconfig -p)"
		if grep -qF "libonnxruntime.so.1 (libc6,x86-64) => $ORT_ROOT/current/lib/" <<<"$cache"; then
			echo "  ok   linker cache resolves libonnxruntime.so.1 via $ORT_ROOT/current/lib"
		else
			echo "  FAIL libonnxruntime.so.1 is not in the linker cache from $ORT_ROOT/current/lib; cache has:"
			grep onnxruntime <<<"$cache" | sed 's/^/         /' || echo "         (no onnxruntime entries)"
			fail=1
		fi
	fi

	((fail)) && die "verification failed for $name"
	info "$name verified"
}

cmd_path() {
	local name
	name="$(resolve_name "${1:-}")"
	echo "$ORT_ROOT/$name/lib/libonnxruntime.so"
}

cmd_remove() {
	local name
	name="$(resolve_name "${1:?usage: remove <name>}")"
	[[ "$(readlink "$ORT_ROOT/current" 2>/dev/null)" == "$name" ]] &&
		die "$name is current; 'use' another install first"
	info "Removing $ORT_ROOT/$name"
	sudo rm -rf -- "${ORT_ROOT:?}/$name"
}

for tool in curl jq sha256sum tar python3 ldd; do
	command -v "$tool" >/dev/null || die "missing required tool: $tool"
done

case "${1:-}" in
install) shift; cmd_install "$@" ;;
use) shift; cmd_use "$@" ;;
list | ls) cmd_list ;;
available) shift; cmd_available "$@" ;;
verify) shift; cmd_verify "$@" ;;
path) shift; cmd_path "$@" ;;
remove | rm) shift; cmd_remove "$@" ;;
-h | --help | help | "") usage ;;
*) die "unknown command: $1 (see --help)" ;;
esac
