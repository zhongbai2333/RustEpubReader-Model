//! C ABI v1 entrypoints for the CSC inference plugin.
//!
//! All exported functions are wrapped in [`std::panic::catch_unwind`] so a
//! Rust panic never unwinds across the FFI boundary into the host.
//!
//! The wire format of `csc_engine_check` is a JSON array:
//!
//! ```json
//! [{"original":"鏅?,"corrected":"娓?,"confidence":0.97,"char_offset":3}]
//! ```
//!
//! The returned pointer must be released by the caller via
//! `csc_string_free`.

mod inference;
mod tokenizer;

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;

use inference::{Correction, Engine};

/// Bumped on incompatible C ABI changes.
const PLUGIN_ABI_VERSION: u32 = 1;

/// Plugin build info 鈥?kept stable across patch releases of ABI v1.
const PLUGIN_VERSION: &str = concat!("csc-plugin/", env!("CARGO_PKG_VERSION"));

/// Opaque handle handed out across FFI.
pub struct CscEngineFfi {
    engine: Option<Engine>,
    last_error: Option<CString>,
    ep_label: CString,
}

impl CscEngineFfi {
    fn new() -> Self {
        Self {
            engine: None,
            last_error: None,
            ep_label: CString::new("none").unwrap(),
        }
    }

    fn set_error(&mut self, msg: impl Into<String>) {
        self.last_error = CString::new(msg.into().replace('\0', "?")).ok();
    }
}

/// Read a C string into Rust. Returns None on null / invalid UTF-8.
unsafe fn cstr_to_string(p: *const c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    Some(CStr::from_ptr(p).to_string_lossy().into_owned())
}

#[no_mangle]
pub extern "C" fn csc_plugin_abi_version() -> u32 {
    PLUGIN_ABI_VERSION
}

#[no_mangle]
pub extern "C" fn csc_engine_version() -> *const c_char {
    // Static C string with terminating NUL 鈥?safe to expose forever.
    static VERSION: std::sync::OnceLock<CString> = std::sync::OnceLock::new();
    VERSION
        .get_or_init(|| CString::new(PLUGIN_VERSION).unwrap())
        .as_ptr()
}

#[no_mangle]
pub extern "C" fn csc_engine_new() -> *mut CscEngineFfi {
    let result = std::panic::catch_unwind(|| Box::into_raw(Box::new(CscEngineFfi::new())));
    result.unwrap_or(std::ptr::null_mut())
}

/// # Safety
/// `engine` must have been returned by [`csc_engine_new`] and not freed.
#[no_mangle]
pub unsafe extern "C" fn csc_engine_free(engine: *mut CscEngineFfi) {
    if engine.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        drop(Box::from_raw(engine));
    }));
}

/// Loads the model + vocab into the engine and selects an execution provider.
///
/// Returns 0 on success, non-zero on failure. Use [`csc_engine_last_error`]
/// to retrieve a human-readable description of the failure.
///
/// # Safety
/// `engine`, `model_path`, `vocab_path`, `ep_hint` must all be valid C strings
/// (non-null UTF-8, NUL-terminated).
#[no_mangle]
pub unsafe extern "C" fn csc_engine_load(
    engine: *mut CscEngineFfi,
    model_path: *const c_char,
    vocab_path: *const c_char,
    ep_hint: *const c_char,
) -> i32 {
    if engine.is_null() {
        return -1;
    }
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let e = &mut *engine;
        let model = match cstr_to_string(model_path) {
            Some(s) => PathBuf::from(s),
            None => {
                e.set_error("model_path is null");
                return -2;
            }
        };
        let vocab = match cstr_to_string(vocab_path) {
            Some(s) => PathBuf::from(s),
            None => {
                e.set_error("vocab_path is null");
                return -2;
            }
        };
        let ep = cstr_to_string(ep_hint).unwrap_or_else(|| "cpu".to_string());

        match Engine::load(&model, &vocab, &ep) {
            Ok(eng) => {
                e.ep_label = CString::new(eng.ep_name()).unwrap_or_else(|_| {
                    CString::new("unknown").unwrap()
                });
                e.engine = Some(eng);
                e.last_error = None;
                0
            }
            Err(err) => {
                e.set_error(format!("{err}"));
                -3
            }
        }
    }));
    match result {
        Ok(rc) => rc,
        Err(_) => {
            if !engine.is_null() {
                (*engine).set_error("panic in csc_engine_load");
            }
            -100
        }
    }
}

/// # Safety
/// `engine` must point to a valid initialized handle.
#[no_mangle]
pub unsafe extern "C" fn csc_engine_last_error(engine: *const CscEngineFfi) -> *const c_char {
    if engine.is_null() {
        return std::ptr::null();
    }
    match &(*engine).last_error {
        Some(c) => c.as_ptr(),
        None => std::ptr::null(),
    }
}

/// # Safety
/// `engine` must point to a valid initialized handle.
#[no_mangle]
pub unsafe extern "C" fn csc_engine_execution_provider(
    engine: *const CscEngineFfi,
) -> *const c_char {
    if engine.is_null() {
        return std::ptr::null();
    }
    (*engine).ep_label.as_ptr()
}

/// Run correction over `text`. Returns a malloc'd JSON string (must be freed
/// by [`csc_string_free`]) or NULL on error / not-loaded.
///
/// # Safety
/// `engine` must be a loaded engine; `text` a valid NUL-terminated UTF-8.
#[no_mangle]
pub unsafe extern "C" fn csc_engine_check(
    engine: *mut CscEngineFfi,
    text: *const c_char,
    threshold: f32,
) -> *mut c_char {
    if engine.is_null() || text.is_null() {
        return std::ptr::null_mut();
    }
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let e = &mut *engine;
        let Some(engine) = e.engine.as_mut() else {
            e.set_error("engine not loaded");
            return std::ptr::null_mut();
        };
        let Some(text) = cstr_to_string(text) else {
            e.set_error("text is null");
            return std::ptr::null_mut();
        };
        let corrections: Vec<Correction> = engine.check(&text, threshold);
        let json = match serde_json::to_string(&corrections) {
            Ok(s) => s,
            Err(err) => {
                e.set_error(format!("serialize: {err}"));
                return std::ptr::null_mut();
            }
        };
        match CString::new(json) {
            Ok(c) => c.into_raw(),
            Err(err) => {
                e.set_error(format!("CString: {err}"));
                std::ptr::null_mut()
            }
        }
    }));
    match result {
        Ok(p) => p,
        Err(_) => {
            if !engine.is_null() {
                (*engine).set_error("panic in csc_engine_check");
            }
            std::ptr::null_mut()
        }
    }
}

/// Reclaim a string previously returned by [`csc_engine_check`] /
/// [`csc_engine_last_error`] cannot use this 鈥?its memory is owned by the
/// engine.
///
/// # Safety
/// `s` must be a pointer returned by `csc_engine_check`, not yet freed.
#[no_mangle]
pub unsafe extern "C" fn csc_string_free(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        drop(CString::from_raw(s));
    }));
}
