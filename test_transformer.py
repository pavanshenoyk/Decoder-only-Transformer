import torch
from model import Transformer  # adjust import to your filename

torch.manual_seed(0)

# --- small config for fast testing ---
d_model = 64
d_ff = 256
head = 4
seq_length = 16
vocab_size = 50
dropout = 0.1
num_layers = 2
batch = 4

model = Transformer(d_model, d_ff, head, seq_length, vocab_size, dropout, num_layers)

# ---------------------------------------------------------------
# Test 1: shape check
# ---------------------------------------------------------------
x = torch.randint(0, vocab_size, (batch, seq_length))
out = model(x)
expected_shape = (batch, seq_length, vocab_size)
assert out.shape == expected_shape, f"Shape mismatch: got {out.shape}, expected {expected_shape}"
print(f"[PASS] output shape {out.shape}")

# ---------------------------------------------------------------
# Test 2: causality check
# changing a future token must NOT change earlier positions' output
# ---------------------------------------------------------------
model.eval()
with torch.no_grad():
    x1 = torch.randint(0, vocab_size, (1, seq_length)).to(model.device)
    x2 = x1.clone()
    x2[0, -1] = (x2[0, -1] + 1) % vocab_size  # change only the last token

    out1 = model(x1)
    out2 = model(x2)

    # all positions except the last should be identical
    same_prefix = torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5)
    assert same_prefix, "Causality broken: changing the last token affected earlier positions"
    print("[PASS] causal masking holds (future token doesn't leak into past positions)")
model.train()

# ---------------------------------------------------------------
# Test 3: tiny overfit test
# model should be able to memorize a handful of (input -> next-token) pairs
# ---------------------------------------------------------------
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()

# fixed tiny "dataset": predict next token given previous tokens
toy_x = torch.randint(0, vocab_size, (2, seq_length)).to(model.device)
toy_y = torch.randint(0, vocab_size, (2, seq_length)).to(model.device)  # random targets, just to test optimization works

losses = []
for step in range(200):
    optimizer.zero_grad()
    logits = model(toy_x)  # (batch, seq_len, vocab_size)
    loss = criterion(logits.view(-1, vocab_size), toy_y.view(-1))
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

print(f"[INFO] loss: start={losses[0]:.4f}  end={losses[-1]:.4f}")
assert losses[-1] < losses[0] * 0.3, "Loss did not drop meaningfully — check gradient flow"
print("[PASS] model can overfit a tiny batch (loss dropped)")

print("\nAll smoke tests passed.")