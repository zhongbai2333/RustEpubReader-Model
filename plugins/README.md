# CSC Inference Plugin

This is the dynamic-library plugin loaded by the **RustEpubReader** main app
to perform Chinese Spelling Correction. It is built and released from this
repository so the host application stays slim — the host downloads the
matching plugin (and the ORT shared library on platforms that need it) on
demand.

## ABI

The C interface is locked at **version 1**. The exported symbols are listed
in [`csc-plugin/include/csc_plugin.h`](csc-plugin/include/csc_plugin.h).
Bumping the ABI version requires a coordinated change in the host repository
(`core/src/csc/plugin.rs::PLUGIN_ABI_VERSION`).

## Layout published to the CDN

```
plugins/v1/<platform>/
    csc_plugin.{dll,so,dylib}        (desktop)
    libonnxruntime.so                (Android only — extracted from AAR)
    libonnxruntime4j_jni.so          (Android only — extracted from AAR)
    csc-plugin-manifest.json
```

`<platform>` is one of:

| Platform                  | Notes                                  |
| ------------------------- | -------------------------------------- |
| `windows-x86_64-cpu`      | CPU-only EP                            |
| `windows-x86_64-directml` | DirectML EP enabled (falls back to CPU)|
| `windows-aarch64-cpu`     | CPU-only EP                            |
| `linux-x86_64`            | CPU-only EP                            |
| `linux-aarch64`           | CPU-only EP                            |
| `macos-x86_64`            | CPU-only EP                            |
| `macos-aarch64`           | CPU-only EP (Apple Silicon)            |
| `android-arm64-v8a`       | ORT runtime libs only                  |
| `android-x86_64`          | ORT runtime libs only                  |

## Building locally

```bash
cd plugins
cargo build --release -p csc-plugin
# DirectML variant on Windows
cargo build --release -p csc-plugin --features directml
```

## CI

Tagging a commit with `plugin-v<version>` triggers
[`.github/workflows/build-plugin.yml`](../.github/workflows/build-plugin.yml),
which builds every platform variant, extracts the Android ORT libs and
publishes a GitHub Release that the CDN mirrors.
