import torch
import torch.nn as nn
import torch.optim as optim
import melee
import os
import pickle
import numpy as np
import signal

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Bayes")))
from dataset_collector import *

class SimpleModel(nn.Module):
	def __init__(self, input_size, output_size):
		super(SimpleModel, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(input_size, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, output_size),
			nn.Sigmoid()  # Ensures outputs are between 0 and 1
		)

	def forward(self, x):
		return self.fc(x)

savefile = "ice_god_falco"
# connect_code = "QHAS#352"
connect_code = ""

character, stage = character_stage[savefile]
# costume = random.randint(0, 4)
costume = 1

minVal = None
maxVal = None

"""
filename = "./pickles/compressed_" + savefile + ".pkl"
if os.path.exists(filename):
	with open(filename, 'rb') as f:
		data = pickle.load(f)
		maxVal = data["maxVal"]
		minVal = data["minVal"]
else:
	data = loadData(savefile)
	minVal, maxVal = data_normalization(data)
	fast_compression(data)

	data["minVal"] = minVal
	data["maxVal"] = maxVal

	with open(filename, 'wb') as f:
		pickle.dump(data, f)
"""

data = loadData(savefile)
minVal, maxVal = data_normalization(data)

N = 32
M = 18
model_path = "./models/" + savefile

print("No model found. Training a new one...")

models = {}

p = 0
keys = list(data["data"].keys())
for i, key in enumerate(keys):
	print(str(p) + "%")

	while (100 * i > p * len(keys)):
		p += 1

	path = model_path + "_" + key + ".pt"

	if os.path.exists(path):
		print("Loading existing model...")
		model = SimpleModel(N, M)
		model.load_state_dict(torch.load(path))
	else:
		model = SimpleModel(N, M)
		criterion = nn.MSELoss()  # Example loss function (modify as needed)
		optimizer = optim.Adam(model.parameters(), lr=0.01)

		X_train = torch.tensor([d["value"].flatten().tolist() for d in data["data"][key]], dtype=torch.float)
		Y_train = torch.tensor([d["input"].to_numpy().flatten().tolist() for d in data["data"][key]], dtype=torch.float)

		epochs = 500
		for epoch in range(epochs):
			optimizer.zero_grad()
			outputs = model(X_train)
			loss = criterion(outputs, Y_train)
			loss.backward()
			optimizer.step()

			if epoch % 50 == 0:
				print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

		torch.save(model.state_dict(), path)
		print("Model saved.")

	model.eval()

	models[key] = model

	del data["data"][key]

console = melee.Console(path="/home/avighna/Downloads/Slippi_Online-x86_64.AppImage")

myPort = 1
opPort = 2

controller = melee.Controller(console=console, port=myPort)
controller_human = melee.Controller(console=console,
									port=opPort,
									type=melee.ControllerType.GCN_ADAPTER)

console.run()
console.connect()

controller.connect()
controller_human.connect()

def signal_handler(sig, frame):
	console.stop()
	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

while True:
	# start = time.time()

	gamestate = console.step()
	if gamestate is None:
		continue

	if console.processingtime * 1000 > 17:
		print("WARNING: Last frame took " + str(console.processingtime*1000) + "ms to process.")

	if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
		if connect_code != "":
			discovered_port = melee.gamestate.port_detector(gamestate, character, costume)
			if (discovered_port != myPort):
				opPort = myPort
				myPort = discovered_port

		key = stringEnumerate(keyInfo(gamestate, myPort, opPort))

		print(key)

		if (key in models):
			model = models[key]
			vDict = valueFn(gamestate, myPort, opPort)
			vDict["value"] = normalize(vDict["value"], minVal, maxVal)

			vDict["input"] = model(torch.from_numpy(vDict["value"]).to(torch.float))

			print(vDict["input"])

			set_controller_state(controller, PickleableControllerState(np_array=vDict["input"].detach().numpy()))
			
		else:
			print("Key loss!")


	else:
		melee.MenuHelper.menu_helper_simple(gamestate,
											controller,
											character,
											stage,
											connect_code,
											costume=costume,
											autostart=True,
											swag=False)

	# end = time.time()

	# print("Frame: " + str(end - start))