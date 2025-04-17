# Decoder-Only Transformer from Scratch - Planning

## 1. Goal
- Implement a functional decoder-only transformer model using PyTorch.
- Focus on understanding the core components and their interactions.
- Optimize for reasonably efficient execution (inference) on CPU (Intel i5-11320H) and potentially leverage integrated GPU (Intel Iris Xe).
- Utilize Huggingface datasets and tokenizer for data processing and preparation.

## 2. Constraints & Considerations
- **Hardware:** 11th Gen Intel Core i5-11320H @ 3.20GHz, 16GB RAM, Intel Iris Xe Graphics.
- **Memory:** 16GB RAM limits the size of the model and batch sizes, especially if training is considered.
- **Compute:** Rely primarily on CPU. Explore potential Iris Xe acceleration later if needed (e.g., via PyTorch with IPEX, ONNX Runtime, or OpenVINO).
- **Dependencies:** PyTorch as the primary framework, Huggingface datasets and tokenizers for data processing.

## 3. Core Architecture
- Input Embedding Layer
- Positional Encoding (Sinusoidal or Learned)
- Multiple Decoder Blocks, each containing:
    - Masked Multi-Head Self-Attention
    - Add & Norm (Layer Normalization)
    - Feed-Forward Network
    - Add & Norm (Layer Normalization)
- Final Linear Layer
- Softmax Layer (for probability distribution)

## 4. Optimization Strategy
- **Model Size:** Start with small dimensions (embedding size, number of heads, feed-forward hidden size, number of layers).
- **Data Types:** Use standard float32 initially. Consider float16 or quantization (e.g., int8) for optimization *after* the basic model works.
- **Implementation:** Focus on clear, understandable PyTorch code. Leverage PyTorch's built-in optimizations and tensor operations. Avoid unnecessary computations.
- **Inference Focus:** Prioritize efficient inference loop. Training efficiency might be secondary given hardware constraints.

## 5. Development Environment
- Python (via `venv` created earlier)
- Libraries: PyTorch (core), Huggingface datasets and tokenizers.
- Version Control: Git (recommended).

## 6. Evaluation
- **Functionality:** Does the model generate coherent (though maybe simple) output?
- **Performance:** Measure inference speed (e.g., tokens per second) on the target CPU.
- **Resource Usage:** Monitor RAM consumption during inference.
