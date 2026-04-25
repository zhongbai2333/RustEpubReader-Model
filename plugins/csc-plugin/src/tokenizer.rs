//! WordPiece tokenizer for the MacBERT-CSC model.
//!
//! Ported from `core/src/csc/tokenizer.rs` of the main repository
//! (commit cea2c8b, before plugin extraction). The behavior must match the
//! Android-side Kotlin tokenizer in `CscEngine.kt` to keep correction offsets
//! consistent across platforms.

use std::path::Path;

/// Maximum sequence length for MacBERT input. Shared with the host app — do
/// **not** change without bumping the plugin ABI version.
pub const MAX_SEQ_LEN: usize = 128;

/// Tokenized output ready for ONNX inference.
pub struct TokenizedInput {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub token_type_ids: Vec<i64>,
    /// Mapping from token index → (char_start, char_end) in original text.
    /// Only valid for non-special tokens (excludes [CLS], [SEP], [PAD]).
    pub offset_mapping: Vec<Option<(usize, usize)>>,
}

/// WordPiece tokenizer backed by the HuggingFace tokenizers crate.
pub struct CscTokenizer {
    inner: tokenizers::Tokenizer,
}

impl CscTokenizer {
    /// Load a tokenizer from a `vocab.txt` file (BERT WordPiece format).
    pub fn from_vocab(vocab_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        use tokenizers::models::wordpiece::WordPiece;

        let wp = WordPiece::from_file(&vocab_path.to_string_lossy())
            .unk_token("[UNK]".into())
            .build()
            .map_err(|e| format!("Failed to load WordPiece vocab: {e}"))?;

        let mut tokenizer = tokenizers::Tokenizer::new(wp);

        // BERT-style normalization (NFC + lowercase + accent stripping for ASCII).
        tokenizer.with_normalizer(Some(tokenizers::normalizers::BertNormalizer::default()));

        // Default BERT pre-tokenizer splits on whitespace + punctuation and
        // emits one token per CJK ideograph, which is exactly what MacBERT-CSC
        // expects.
        tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::sequence::Sequence::new(
            vec![tokenizers::pre_tokenizers::bert::BertPreTokenizer.into()],
        )));

        Ok(Self { inner: tokenizer })
    }

    /// Tokenize a single sentence. Adds `[CLS]` / `[SEP]` and pads to
    /// [`MAX_SEQ_LEN`] so the ONNX session always receives a fixed-shape input.
    pub fn encode(&self, text: &str) -> Result<TokenizedInput, Box<dyn std::error::Error>> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| format!("Tokenization error: {e}"))?;

        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mut attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let mut token_type_ids: Vec<i64> = encoding
            .get_type_ids()
            .iter()
            .map(|&t| t as i64)
            .collect();

        let raw_offsets = encoding.get_offsets();
        let special_mask = encoding.get_special_tokens_mask();

        // HuggingFace tokenizers return *byte* offsets, but downstream
        // consumers want *char* offsets to index into the original sentence
        // safely. Build a precomputed byte→char map.
        let byte_to_char: Vec<usize> = {
            let char_count = text.chars().count();
            let mut map = vec![char_count; text.len() + 1];
            for (ci, (bi, _)) in text.char_indices().enumerate() {
                map[bi] = ci;
            }
            map[text.len()] = char_count;
            map
        };

        let mut offset_mapping: Vec<Option<(usize, usize)>> = raw_offsets
            .iter()
            .zip(special_mask.iter())
            .map(|(&(s, e), &is_special)| {
                if is_special != 0 {
                    None
                } else {
                    let cs = byte_to_char.get(s).copied().unwrap_or(0);
                    let ce = byte_to_char
                        .get(e)
                        .copied()
                        .unwrap_or(text.chars().count());
                    Some((cs, ce))
                }
            })
            .collect();

        if input_ids.len() > MAX_SEQ_LEN {
            input_ids.truncate(MAX_SEQ_LEN);
            attention_mask.truncate(MAX_SEQ_LEN);
            token_type_ids.truncate(MAX_SEQ_LEN);
            offset_mapping.truncate(MAX_SEQ_LEN);
            // Patch the last token to be [SEP] so the model still sees a
            // properly framed sequence.
            if let Some(sep_id) = self.inner.token_to_id("[SEP]") {
                if let Some(last) = input_ids.last_mut() {
                    *last = sep_id as i64;
                }
                if let Some(last) = offset_mapping.last_mut() {
                    *last = None;
                }
            }
        }

        let pad_id = self.inner.token_to_id("[PAD]").unwrap_or(0) as i64;
        while input_ids.len() < MAX_SEQ_LEN {
            input_ids.push(pad_id);
            attention_mask.push(0);
            token_type_ids.push(0);
            offset_mapping.push(None);
        }

        Ok(TokenizedInput {
            input_ids,
            attention_mask,
            token_type_ids,
            offset_mapping,
        })
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    pub fn unk_id(&self) -> Option<u32> {
        self.inner.token_to_id("[UNK]")
    }
}
