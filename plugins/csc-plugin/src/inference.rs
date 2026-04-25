//! Inference engine — runs the MacBERT-CSC ONNX model and applies the same
//! post-processing filters as the original `core/src/csc/mod.rs` of the host
//! repository (commit cea2c8b, before plugin extraction).

use std::path::Path;

use crate::tokenizer::{CscTokenizer, MAX_SEQ_LEN};

/// One detected correction. Serializable to the JSON wire format consumed by
/// the host application.
#[derive(serde::Serialize)]
pub struct Correction {
    pub original: String,
    pub corrected: String,
    pub confidence: f32,
    pub char_offset: usize,
}

pub struct Engine {
    session: ort::session::Session,
    tokenizer: CscTokenizer,
    ep_name: &'static str,
}

impl Engine {
    pub fn ep_name(&self) -> &'static str {
        self.ep_name
    }

    /// Build a session by trying execution providers in order, falling back
    /// to CPU. `ep_hint` is "cpu" / "directml" / "cuda" — unknown values fall
    /// back to CPU.
    pub fn load(
        model_path: &Path,
        vocab_path: &Path,
        ep_hint: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if !model_path.exists() {
            return Err(format!("model not found: {}", model_path.display()).into());
        }
        if !vocab_path.exists() {
            return Err(format!("vocab not found: {}", vocab_path.display()).into());
        }

        let (session, ep_name) = create_session(model_path, ep_hint)?;
        let tokenizer = CscTokenizer::from_vocab(vocab_path)?;

        Ok(Self {
            session,
            tokenizer,
            ep_name,
        })
    }

    /// Run correction over a (potentially multi-sentence) text. Sentences are
    /// split on Chinese terminators + newline, then each is fed to the model
    /// independently. Char offsets are returned relative to the input text.
    pub fn check(&mut self, text: &str, threshold: f32) -> Vec<Correction> {
        let mut all = Vec::new();

        let mut cumulative_offset: usize = 0;
        let mut current = String::new();
        let mut seg_char_start: usize = 0;

        for ch in text.chars() {
            current.push(ch);
            if matches!(ch, '。' | '！' | '？' | '；' | '\n') {
                let trimmed = current.trim();
                if !trimmed.is_empty() {
                    let leading_ws = current.chars().take_while(|c| c.is_whitespace()).count();
                    if let Ok(corrections) = self.infer_sentence(trimmed, threshold) {
                        for mut c in corrections {
                            c.char_offset += seg_char_start + leading_ws;
                            all.push(c);
                        }
                    }
                }
                cumulative_offset += current.chars().count();
                seg_char_start = cumulative_offset;
                current.clear();
            }
        }

        let trimmed = current.trim();
        if !trimmed.is_empty() {
            let leading_ws = current.chars().take_while(|c| c.is_whitespace()).count();
            if let Ok(corrections) = self.infer_sentence(trimmed, threshold) {
                for mut c in corrections {
                    c.char_offset += seg_char_start + leading_ws;
                    all.push(c);
                }
            }
        }

        all
    }

    fn infer_sentence(
        &mut self,
        sentence: &str,
        threshold: f32,
    ) -> Result<Vec<Correction>, Box<dyn std::error::Error>> {
        let unk_id = self.tokenizer.unk_id().unwrap_or(100) as i64;
        let encoded = self.tokenizer.encode(sentence)?;
        let seq_len = MAX_SEQ_LEN;

        let input_ids_val =
            ort::value::Tensor::from_array(([1usize, seq_len], encoded.input_ids.clone()))?;
        let attention_mask_val =
            ort::value::Tensor::from_array(([1usize, seq_len], encoded.attention_mask.clone()))?;
        let token_type_ids_val =
            ort::value::Tensor::from_array(([1usize, seq_len], encoded.token_type_ids.clone()))?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids_val,
            "attention_mask" => attention_mask_val,
            "token_type_ids" => token_type_ids_val,
        ]?)?;

        // Output shape: [1, seq_len, vocab_size]. The view is contiguous in
        // memory (model has fixed input shape), so we can take a flat slice.
        let logits = outputs[0].try_extract_tensor::<f32>()?;
        let shape: Vec<usize> = logits.shape().to_vec();
        let logits_flat: &[f32] = logits
            .as_slice()
            .ok_or("non-contiguous logits tensor")?;
        let vocab_size = shape[2];

        let chars: Vec<char> = sentence.chars().collect();
        let mut corrections = Vec::new();

        // Predicted token must beat the original by at least this margin in
        // softmax space; suppresses uncertain corrections.
        const MIN_MARGIN: f32 = 0.20;

        for (pos, &input_id) in encoded.input_ids.iter().enumerate() {
            // Skip [CLS] / [SEP] / [PAD]
            if encoded.offset_mapping[pos].is_none() || encoded.attention_mask[pos] == 0 {
                continue;
            }
            // Skip OOV — model will always "correct" [UNK] to something.
            if input_id == unk_id {
                continue;
            }

            let base = pos * vocab_size;
            let logit_slice = &logits_flat[base..base + vocab_size];

            let predicted_id = logit_slice
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            if predicted_id as i64 == input_id {
                continue;
            }

            // Softmax — but only need P(predicted) and P(original).
            let max_logit = logit_slice
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logit_slice.iter().map(|&x| (x - max_logit).exp()).sum();
            let p_predicted = (logit_slice[predicted_id] - max_logit).exp() / exp_sum;
            let p_original = (logit_slice[input_id as usize] - max_logit).exp() / exp_sum;

            if p_predicted < threshold {
                continue;
            }
            if p_predicted - p_original < MIN_MARGIN {
                continue;
            }

            let Some((char_start, _char_end)) = encoded.offset_mapping[pos] else {
                continue;
            };
            let original_char: String = if char_start < chars.len() {
                chars[char_start].to_string()
            } else {
                self.tokenizer
                    .id_to_token(input_id as u32)
                    .unwrap_or_default()
            };

            // Filter: only correct CJK ideographs.
            if let Some(ch) = original_char.chars().next() {
                if !is_cjk_char(ch) {
                    continue;
                }
            }
            // Filter: protect grammatical particles & function words.
            if is_protected_char(&original_char) {
                continue;
            }

            let corrected_char = self
                .tokenizer
                .id_to_token(predicted_id as u32)
                .unwrap_or_default();
            if corrected_char == "[UNK]" || corrected_char == original_char {
                continue;
            }
            if corrected_char.starts_with("##") {
                continue;
            }
            if is_confused_pair(&original_char, &corrected_char) {
                continue;
            }

            // Reject if the correction would create a duplicate adjacent char.
            if let Some(corr_ch) = corrected_char.chars().next() {
                if char_start > 0 && chars[char_start - 1] == corr_ch {
                    continue;
                }
                if char_start + 1 < chars.len() && chars[char_start + 1] == corr_ch {
                    continue;
                }
            }

            corrections.push(Correction {
                original: original_char,
                corrected: corrected_char,
                confidence: p_predicted,
                char_offset: char_start,
            });
        }

        Ok(corrections)
    }
}

fn create_session(
    model_path: &Path,
    ep_hint: &str,
) -> Result<(ort::session::Session, &'static str), Box<dyn std::error::Error>> {
    use ort::session::Session;
    use ort::session::builder::GraphOptimizationLevel;

    let hint = ep_hint.to_ascii_lowercase();
    let try_order: &[&str] = match hint.as_str() {
        "directml" => &["directml", "cpu"],
        "cuda" => &["cuda", "cpu"],
        _ => &["cpu"],
    };

    for &ep in try_order {
        match build_session(model_path, ep) {
            Ok(session) => {
                let label = match ep {
                    "directml" => "DirectML",
                    "cuda" => "CUDA",
                    _ => "CPU",
                };
                return Ok((session, label));
            }
            Err(e) => {
                eprintln!("[csc-plugin] EP `{ep}` unavailable: {e}");
            }
        }
    }

    // Final CPU fallback (always supported).
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(2)?
        .commit_from_file(model_path)?;
    Ok((session, "CPU"))
}

#[allow(unused_variables)]
fn build_session(
    model_path: &Path,
    ep: &str,
) -> Result<ort::session::Session, Box<dyn std::error::Error>> {
    use ort::session::Session;
    use ort::session::builder::GraphOptimizationLevel;

    let mut builder = Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?;

    match ep {
        #[cfg(feature = "directml")]
        "directml" => {
            use ort::execution_providers::DirectMLExecutionProvider;
            builder = builder.with_execution_providers([DirectMLExecutionProvider::default().build()])?;
        }
        #[cfg(feature = "cuda")]
        "cuda" => {
            use ort::execution_providers::CUDAExecutionProvider;
            builder = builder.with_execution_providers([CUDAExecutionProvider::default().build()])?;
        }
        "cpu" => {
            builder = builder.with_intra_threads(2)?;
        }
        other => {
            return Err(format!("EP `{other}` not compiled into this plugin").into());
        }
    }

    Ok(builder.commit_from_file(model_path)?)
}

/// Known homophone groups MacBERT cannot reliably distinguish.
fn is_confused_pair(original: &str, corrected: &str) -> bool {
    const GROUPS: &[&[&str]] = &[
        &["他", "她", "它", "牠", "祂"],
        &["的", "得", "地"],
        &["做", "作"],
        &["哪", "那"],
        &["在", "再"],
    ];
    let a = original.trim();
    let b = corrected.trim();
    GROUPS
        .iter()
        .any(|group| group.contains(&a) && group.contains(&b))
}

fn is_cjk_char(ch: char) -> bool {
    let cp = ch as u32;
    matches!(
        cp,
        0x4E00..=0x9FFF
        | 0x3400..=0x4DBF
        | 0x20000..=0x2A6DF
        | 0x2A700..=0x2B73F
        | 0x2B740..=0x2B81F
        | 0xF900..=0xFAFF
        | 0x2F800..=0x2FA1F
    )
}

/// Grammatical particles & high-frequency function words that are almost
/// never genuine misspellings. The model frequently mispredicts them because
/// they carry little semantic weight.
fn is_protected_char(s: &str) -> bool {
    const PROTECTED: &[&str] = &[
        // Modal particles
        "吗", "吧", "呢", "啊", "呀", "哇", "哦", "嗯", "喔", "噢", "啦", "嘛", "咯", "喽", "嘞",
        "罢", "咧",
        // Structural particles
        "的", "得", "地",
        // Aspect particles
        "了", "过", "着",
        // Common function words
        "么", "个", "们", "这", "那", "就", "都", "也", "又", "才", "把", "被", "让", "给", "向",
        "往", "从", "到", "为", "而", "且", "或", "与", "及",
        // Pronouns
        "我", "你", "他", "她", "它", "谁", "啥",
        // Demonstratives & measure words
        "这", "那", "哪", "几", "多", "些",
    ];
    PROTECTED.contains(&s.trim())
}
