/*
 * csc_plugin.h — C ABI v1 for the Chinese Spelling Correction inference plugin.
 *
 * The plugin is built and distributed from the RustEpubReader-Model
 * repository and downloaded by the host application on demand. The host
 * dlopens the platform-specific dynamic library and resolves the symbols
 * declared below.
 *
 * All functions are thread-compatible (a single CscEngine handle MUST NOT
 * be used concurrently from multiple threads, but distinct handles may be
 * used in parallel). All heap-allocated strings returned by the plugin must
 * be released with `csc_string_free` to avoid leaks.
 */

#ifndef CSC_PLUGIN_H
#define CSC_PLUGIN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque engine handle. */
typedef struct CscEngineFfi CscEngineFfi;

/* Returns the ABI version implemented by this plugin. The host fails fast
 * if the value differs from its compiled-in expected version. Currently 1. */
uint32_t csc_plugin_abi_version(void);

/* Static, NUL-terminated build-info string (e.g. "csc-plugin/1.0.0").
 * The pointer is valid for the lifetime of the loaded library. */
const char* csc_engine_version(void);

/* Allocate a new uninitialized engine. Returns NULL on out-of-memory. */
CscEngineFfi* csc_engine_new(void);

/* Free an engine previously returned by csc_engine_new. */
void csc_engine_free(CscEngineFfi* engine);

/* Load the model + vocab and select an execution provider.
 *
 * `ep_hint` is one of:
 *   "cpu"      — CPU-only (always works)
 *   "directml" — DirectML on Windows (falls back to CPU if unavailable)
 *   "cuda"     — CUDA on Linux/Windows x86_64 (falls back to CPU)
 *
 * Returns 0 on success, non-zero on failure. Use csc_engine_last_error to
 * retrieve a human-readable error message. */
int32_t csc_engine_load(
    CscEngineFfi* engine,
    const char* model_path,
    const char* vocab_path,
    const char* ep_hint
);

/* Returns the last error string set on this engine, or NULL if none.
 * The pointer is owned by the engine — DO NOT pass to csc_string_free. */
const char* csc_engine_last_error(const CscEngineFfi* engine);

/* Returns the human-readable name of the active execution provider
 * (e.g. "CPU", "DirectML", "CUDA"). Owned by the engine. */
const char* csc_engine_execution_provider(const CscEngineFfi* engine);

/* Run correction over `text` and return a JSON array of corrections:
 *   [
 *     {"original":"晴","corrected":"清","confidence":0.97,"char_offset":3}
 *   ]
 *
 * `threshold` is the minimum softmax probability for a correction to be
 * emitted (typical: 0.80–0.97).
 *
 * Returns NULL on error or if the engine is not loaded. The returned
 * pointer must be freed with csc_string_free. */
char* csc_engine_check(
    CscEngineFfi* engine,
    const char* text,
    float threshold
);

/* Free a heap-allocated string previously returned by csc_engine_check. */
void csc_string_free(char* s);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CSC_PLUGIN_H */
