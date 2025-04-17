# Decoder-Only Transformer from Scratch - Tasks (PyTorch Implementation)

## Phase 1: Project Setup & Core Components Implementation

- [X] **Task 1.1:** Set up project directory structure (`src/`, `tests/`, `data/`, etc.).
- [X] **Task 1.2:** Install and configure dependencies (PyTorch, Huggingface datasets, tokenizers).
- [X] **Task 1.3:** Implement Input Embedding Layer using PyTorch's nn.Embedding.
- [X] **Task 1.4:** Implement Positional Encoding (e.g., Sinusoidal or Learned).
- [X] **Task 1.5:** Implement Scaled Dot-Product Attention (with masking) using PyTorch tensor operations.
- [ ] **Task 1.6:** Implement Multi-Head Self-Attention (using Scaled Dot-Product Attention).
- [ ] **Task 1.7:** Use PyTorch's nn.LayerNorm for Layer Normalization.
- [ ] **Task 1.8:** Implement Position-wise Feed-Forward Network using PyTorch's nn.Linear.
- [ ] **Task 1.9:** Assemble a single Decoder Block as a PyTorch nn.Module.

## Phase 2: Model Assembly & Basic Functionality

- [ ] **Task 2.1:** Stack multiple Decoder Blocks to form the main transformer body as a PyTorch nn.ModuleList.
- [ ] **Task 2.2:** Implement the final Linear output layer using nn.Linear.
- [ ] **Task 2.3:** Use PyTorch's F.softmax for the output probability distribution.
- [ ] **Task 2.4:** Combine all parts into a complete Decoder model class (nn.Module).
- [ ] **Task 2.5:** Create a basic inference function (e.g., greedy decoding) using PyTorch operations.
- [ ] **Task 2.6:** Test with dummy/random data to ensure shapes and basic flow work.
- [ ] **Task 2.7:** Implement model checkpointing using PyTorch's save/load functionality.

## Phase 3: Huggingface Integration & Training Setup

- [ ] **Task 3.1:** Set up a Huggingface dataset for training (e.g., small text corpus).
- [ ] **Task 3.2:** Implement data preprocessing using Huggingface tokenizers.
- [ ] **Task 3.3:** Create PyTorch DataLoader for efficient batch processing.
- [ ] **Task 3.4:** Use PyTorch's nn.CrossEntropyLoss for the loss function.
- [ ] **Task 3.5:** Implement a training loop with PyTorch's optimizer (e.g., Adam).
- [ ] **Task 3.6:** Add learning rate scheduling using PyTorch's lr_scheduler.
- [ ] **Task 3.7:** Implement gradient clipping and other training stabilization techniques.

## Phase 4: Optimization & Refinement

- [ ] **Task 4.1:** Profile inference speed and memory usage using PyTorch profiler.
- [ ] **Task 4.2:** Experiment with model size parameters (layers, dimensions).
- [ ] **Task 4.3:** Implement PyTorch's dynamic quantization for improved inference efficiency.
- [ ] **Task 4.4:** Explore Intel Iris Xe acceleration using PyTorch IPEX.
- [ ] **Task 4.5:** Implement mixed precision training using PyTorch's AMP (Automatic Mixed Precision).
- [ ] **Task 4.6:** Optimize memory usage with gradient checkpointing if needed.

## Phase 5: Documentation & Testing

- [ ] **Task 5.1:** Add docstrings and comments to the code.
- [ ] **Task 5.2:** Write unit tests for key components using PyTorch's testing utilities.
- [ ] **Task 5.3:** Create a model usage example notebook.
- [ ] **Task 5.4:** Document model architecture, training process, and performance metrics.
- [ ] **Task 5.5:** Update `planning.md` and `task.md` as needed.
