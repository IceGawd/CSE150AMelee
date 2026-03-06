import torch
import torch.nn as nn
import torch.optim as optim
import math
import random

# ---- Hyperparameters ----
BATCH_SIZE = 512
EPOCHS = 2000
LR = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Generate training data ----
def generate_batch(batch_size):
    r = torch.rand(batch_size, 1)  # radius in [0,1]
    theta = torch.rand(batch_size, 1) * 2 * math.pi  # angle in [0, 2pi]

    x = r * torch.cos(theta)
    y = r * torch.sin(theta)

    inputs = torch.cat([r, theta], dim=1)
    targets = torch.cat([x, y], dim=1)
    return inputs.to(DEVICE), targets.to(DEVICE)

# ---- Simple MLP ----
class PolarToCartesian(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )

    def forward(self, x):
        return self.net(x)

model = PolarToCartesian().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# ---- Training loop ----
for epoch in range(EPOCHS):
    inputs, targets = generate_batch(BATCH_SIZE)

    preds = model(inputs)
    loss = loss_fn(preds, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# ---- Test edge case near discontinuity ----
test_angles = torch.tensor([[0.0], [math.pi / 4], [math.pi / 3], [math.pi / 2], [math.pi], [2*math.pi - 0.001]]).to(DEVICE)
test_r = torch.ones_like(test_angles)
test_input = torch.cat([test_r, test_angles], dim=1)

with torch.no_grad():
    pred = model(test_input)
    true_x = test_r * torch.cos(test_angles)
    true_y = test_r * torch.sin(test_angles)
    true = torch.cat([true_x, true_y], dim=1)

print("\nEdge Case Test:")
print("Predicted:\n", pred.cpu())
print("True:\n", true.cpu())