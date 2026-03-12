import torch
import torch.nn as nn
import torch.optim as optim
import math
import random

# ---- Hyperparameters ----
BATCH_SIZE = 100
EPOCHS = 100
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# INPUT_SIZE = 2
# OUTPUT_SIZE = 2
INPUT_SIZE = 10
OUTPUT_SIZE = 1

# ---- Generate training data ----
def generate_C2P_batch(batch_size, theta_min=0, theta_max=2*math.pi):
	r = torch.rand(batch_size, 1)  # radius in [0,1]
	theta = torch.rand(batch_size, 1) * (theta_max - theta_min) + theta_min  # angle in [0, 2pi]

	x = r * torch.cos(theta)
	y = r * torch.sin(theta)

	inputs = torch.cat([r, theta], dim=1)
	targets = torch.cat([x, y], dim=1)
	return inputs.to(DEVICE), targets.to(DEVICE)

def generate_sine_batch(batch_size):
	theta = torch.rand(batch_size, 1) * 2 * math.pi

	inputs = torch.cat([torch.sin(theta - (INPUT_SIZE - i) / INPUT_SIZE) for i in range(INPUT_SIZE)], dim=1)
	targets = torch.cat([torch.sin(theta)], dim=1)
	return inputs.to(DEVICE), targets.to(DEVICE)

def generate_overfit(batch_size):
	values = torch.round(torch.rand(batch_size, 1))
	keys = torch.cat([torch.randn((batch_size, INPUT_SIZE - 1)), values], dim=1)

	return keys.to(DEVICE), values.to(DEVICE)

# ---- Test the model on deterministic angles ----
def analysis(x, preds, true_targets):
	print("x | pred | true | error")
	for i in range(len(x)):
		pred = preds[i].item()
		true = true_targets[i].item()
		err = abs(pred - true)
		print(f"{x[i][0].item():.3f} | {pred:.4f} | {true:.4f} | {err:.4f}")


def test_sine_model(model, num_tests=10):
	model.eval()

	# evenly spaced angles across the circle
	thetas = torch.linspace(0, 2 * math.pi, num_tests).unsqueeze(1).to(DEVICE)

	inputs = torch.cat(
		[torch.sin(thetas - (INPUT_SIZE - i) / INPUT_SIZE) for i in range(INPUT_SIZE)],
		dim=1
	)

	true_targets = torch.sin(thetas)

	with torch.no_grad():
		preds = model(inputs)

	print("\n--- Sine Test ---")
	analysis(thetas, preds, true_targets)

# ---- Simple MLP ----
class SimpleMLP(nn.Module):
	def __init__(self, input_size, output_size):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_size, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, output_size)
		)

	def forward(self, x):
		return self.net(x)

model = SimpleMLP(INPUT_SIZE, OUTPUT_SIZE).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
# loss_fn = nn.BCEWithLogitsLoss()

print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# ---- Training loop ----

inputs, targets = generate_overfit(BATCH_SIZE)
for epoch in range(EPOCHS):
	# piece = int(random.random() * EPOCHS)
	# inputs, targets = generate_C2P_batch(BATCH_SIZE)
	# inputs, targets = generate_C2P_batch(BATCH_SIZE, theta_min=2*math.pi*piece/EPOCHS, theta_max=2*math.pi*(piece + 1)/EPOCHS)
	# inputs, targets = generate_sine_batch(BATCH_SIZE)

	preds = model(inputs)

	loss = loss_fn(preds, targets)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	if epoch % (EPOCHS // 10) == 0:
		print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# # ---- Test edge case near discontinuity ----
# test_angles = torch.tensor([[0.0], [math.pi / 4], [math.pi / 3], [math.pi / 2], [math.pi], [2*math.pi - 0.001]]).to(DEVICE)
# test_r = torch.ones_like(test_angles)
# test_input = torch.cat([test_r, test_angles], dim=1)

# with torch.no_grad():
#     pred = model(test_input)
#     true_x = test_r * torch.cos(test_angles)
#     true_y = test_r * torch.sin(test_angles)
#     true = torch.cat([true_x, true_y], dim=1)

# print("\nEdge Case Test:")
# print("Predicted:\n", pred.cpu())
# print("True:\n", true.cpu())

# test_sine_model(model, num_tests=9)

preds = 1 / (1 + torch.exp(-model(inputs[0:10])))
analysis(inputs[0:10], preds, targets[0:10])