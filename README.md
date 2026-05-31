# makemore — Indian Name Generator 🇮🇳

> *Making more Indian names, one character at a time.*

A character-level language model that generates new, realistic-sounding **Indian names** — trained purely on a dataset of 114,240 Indian names. Inspired by [Andrej Karpathy's makemore](https://github.com/karpathy/makemore) series, this project walks through building progressively more powerful autoregressive models from scratch using PyTorch.

---

## What is makemore?

`makemore` takes a list of names and learns to **make more of them**. It treats every name as a sequence of characters and trains a model to predict the next character given the previous context. At inference time, you sample from the model character by character until it outputs an end token — giving you a brand-new, never-before-seen name.

This repo applies that idea to **Indian names** — covering a rich diversity of Hindu, Muslim, Sikh, Christian, and regional names from across the subcontinent.

---

## Dataset

- **File:** `names.txt`
- **Size:** 114,240 unique Indian names
- **Format:** One lowercase name per line
- **Examples:** `lakshay`, `ganeshkumar`, `sanjivni`, `soumava`, `thayashana`, `yuktvaa`

The dataset spans names from different Indian languages and communities, giving the model exposure to a wide variety of phonetic patterns, prefixes, and suffixes characteristic of Indian naming conventions.

---

## Model Progression

Following Karpathy's lecture series, models are built up in complexity:

| # | Model | Description |
|---|-------|-------------|
| 1 | **Bigram** | Counts character pairs; simplest baseline |
| 2 | **MLP** | Fixed context window → hidden layer → output (Bengio et al. 2003) |
| 3 | **RNN** | Recurrent network with hidden state |
| 4 | **LSTM** | Long Short-Term Memory; better gradient flow |
| 5 | **WaveNet** | Hierarchical dilated convolutions (DeepMind 2016) |
| 6 | **Transformer** | Self-attention based model (GPT-style) |

Each model is implemented from scratch in pure PyTorch — no `nn.RNN`, no HuggingFace, no shortcuts.

---

## Getting Started

### Prerequisites

```bash
pip install torch numpy matplotlib
```

### Generate Names

```python
# Example (once a model is trained)
model.eval()
for _ in range(10):
    out = []
    context = [0] * block_size
    while True:
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        if ix == 0:
            break
        out.append(itos[ix])
    print(''.join(out))
```

Sample outputs might look like: `arjunveer`, `priyasha`, `devnath`, `kaavyaa`, `shivaan`

---

## Project Structure

```
makemore/
├── names.txt       # 114,240 Indian names (training data)
├── README.md       # You are here
└── makemore.py     # Model implementations
```

---

## References

- [Andrej Karpathy — makemore (GitHub)](https://github.com/karpathy/makemore)
- [Neural Probabilistic Language Model — Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- [WaveNet — van den Oord et al. 2016](https://arxiv.org/abs/1609.03499)
- [Attention Is All You Need — Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)
- [Karpathy's makemore lecture playlist (YouTube)](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

---

## Acknowledgements

Built by following Andrej Karpathy's legendary *Neural Networks: Zero to Hero* series. The Indian names dataset was curated to explore how character-level models capture the phonetics and structure of a culturally rich and linguistically diverse naming tradition.
