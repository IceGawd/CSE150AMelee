import torch
import torch.nn as nn
import torch.optim as optim
import math
import random

# ---- Hyperparameters ----
BATCH_SIZE = 100
EPOCHS = 1000
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Generate training data ----
def generate_batch(batch_size, theta_min=0, theta_max=2*math.pi):
    r = torch.rand(batch_size, 1)  # radius in [0,1]
    theta = torch.rand(batch_size, 1) * (theta_max - theta_min) + theta_min  # angle in [0, 2pi]

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
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

model = PolarToCartesian().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# ---- Training loop ----
for epoch in range(EPOCHS):
    # piece = int(random.random() * EPOCHS)
    inputs, targets = generate_batch(BATCH_SIZE)
    # inputs, targets = generate_batch(BATCH_SIZE, theta_min=2*math.pi*piece/EPOCHS, theta_max=2*math.pi*(piece + 1)/EPOCHS)

    preds = model(inputs)

    loss = loss_fn(preds, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % (EPOCHS // 10) == 0:
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