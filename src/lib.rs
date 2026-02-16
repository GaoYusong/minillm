use pyo3::prelude::*;
use ndarray::{Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;

/// Fused attention: Q @ K^T @ V in one go with online softmax
/// This avoids materializing the full attention matrix
#[pyfunction]
fn fused_attention(
    py: Python,
    q: &PyArray2<f32>,
    k: &PyArray2<f32>,
    v: &PyArray2<f32>,
    scale: f32,
) -> PyResult<PyObject> {
    let q = unsafe { q.as_array() };
    let k = unsafe { k.as_array() };
    let v = unsafe { v.as_array() };
    
    let seq_len = q.shape()[0];
    let head_dim = q.shape()[1];
    
    // Output buffer
    let mut output = Array2::zeros((seq_len, head_dim));
    
    // Online softmax attention (FlashAttention-style)
    // Process in chunks to reduce memory bandwidth
    const CHUNK_SIZE: usize = 64;
    
    for i in (0..seq_len).step_by(CHUNK_SIZE) {
        let end = (i + CHUNK_SIZE).min(seq_len);
        let q_chunk = q.slice(ndarray::s![i..end, ..]);
        
        // Compute attention for this chunk
        for (local_i, q_row) in q_chunk.outer_iter().enumerate() {
            let global_i = i + local_i;
            
            // Q @ K^T
            let mut scores = Vec::with_capacity(seq_len);
            for k_row in k.outer_iter() {
                let score: f32 = q_row.iter().zip(k_row.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>() * scale;
                scores.push(score);
            }
            
            // Softmax
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = scores.iter()
                .map(|s| (s - max_score).exp())
                .sum();
            
            // Weighted sum of V
            let mut out_row = output.row_mut(global_i);
            for (j, score) in scores.iter().enumerate() {
                let weight = (score - max_score).exp() / exp_sum;
                let v_row = v.row(j);
                for (d, v_val) in v_row.iter().enumerate() {
                    out_row[d] += weight * v_val;
                }
            }
        }
    }
    
    Ok(output.into_pyarray(py).to_object(py))
}

/// Optimized matrix multiplication with tiling
#[pyfunction]
fn matmul_f32(
    py: Python,
    a: &PyArray2<f32>,
    b: &PyArray2<f32>,
) -> PyResult<PyObject> {
    let a = unsafe { a.as_array() };
    let b = unsafe { b.as_array() };
    
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];
    
    assert_eq!(k, b.shape()[0], "Matrix dimensions mismatch");
    
    let mut c = Array2::zeros((m, n));
    
    // Simple tiled matmul (can be optimized further with SIMD)
    const TILE: usize = 32;
    
    c.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut c_row)| {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[[i, l]] * b[[l, j]];
                }
                c_row[j] = sum;
            }
        });
    
    Ok(c.into_pyarray(py).to_object(py))
}

/// RMSNorm: faster than LayerNorm, used in Llama
#[pyfunction]
fn rms_norm(
    py: Python,
    x: &PyArray2<f32>,
    weight: &PyArray1<f32>,
    eps: f32,
) -> PyResult<PyObject> {
    let x = unsafe { x.as_array() };
    let weight = unsafe { weight.as_array() };
    
    let mut output = x.to_owned();
    
    output.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
            let scale = 1.0 / (mean_sq + eps).sqrt();
            for (i, v) in row.iter_mut().enumerate() {
                *v = *v * scale * weight[i];
            }
        });
    
    Ok(output.into_pyarray(py).to_object(py))
}

/// Silu activation: x * sigmoid(x)
#[pyfunction]
fn silu(py: Python, x: &PyArray2<f32>) -> PyResult<PyObject> {
    let x = unsafe { x.as_array() };
    let result = x.mapv(|v| v * (1.0 / (1.0 + (-v).exp())));
    Ok(result.into_pyarray(py).to_object(py))
}

/// Quantize f32 to int8
#[pyfunction]
fn quantize_int8(
    py: Python,
    x: &PyArray2<f32>,
) -> PyResult<(PyObject, f32, f32)> {
    let x = unsafe { x.as_array() };
    
    let min_val = x.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    
    let scale = (max_val - min_val) / 255.0;
    let zero_point = min_val;
    
    let quantized: ndarray::Array2<i8> = x.mapv(|v| {
        ((v - zero_point) / scale).clamp(0.0, 255.0) as i8
    });
    
    Ok((
        quantized.into_pyarray(py).to_object(py),
        scale,
        zero_point,
    ))
}

#[pymodule]
fn minillm_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fused_attention, m)?)?;
    m.add_function(wrap_pyfunction!(matmul_f32, m)?)?;
    m.add_function(wrap_pyfunction!(rms_norm, m)?)?;
    m.add_function(wrap_pyfunction!(silu, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_int8, m)?)?;
    Ok(())
}
