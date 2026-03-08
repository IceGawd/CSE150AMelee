import torch
import melee
import numpy as np
import signal
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Bayes")))
from train import (
	CONFIG,
	processGamestate,
	loadNNData,
	count_parameters
)

device = "cuda" if torch.cuda.is_available() else "cpu"

savefile = "ice_god_samus"

character, stage = character_stage[savefile]
costume = 1
connect_code = ""

# ----------------------------
# Load model
# ----------------------------

data = loadNNData(savefile)

model = data["data"]["model"]
lstm = data["data"]["lstm"]
embed = data["data"]["embed"]

model.eval()
lstm.eval()
embed.eval()

total_params = (
	count_parameters(model)
	+ count_parameters(lstm)
	+ count_parameters(embed)
)

print("Total parameters:", total_params)
print("Frames trained on:", data["data"]["frames"])

# ----------------------------
# Slippi setup
# ----------------------------

console = melee.Console(path="/home/avighna/Downloads/Slippi_Online-x86_64_3.5.2.AppImage")

myPort = 1
opPort = 2

controller = melee.Controller(console=console, port=myPort)
controller_human = melee.Controller(
	console=console,
	port=opPort,
	type=melee.ControllerType.GCN_ADAPTER
)

console.run()
console.connect()

controller.connect()
controller_human.connect()

# ----------------------------
# Runtime state
# ----------------------------

frame_history = []
input_history = []
hidden = None

full_state_size = (
	CONFIG["frame_feature_size"]
	+ CONFIG["action_embed_size"]
	+ CONFIG["lstm_hidden"]
)

# ----------------------------
# Clean exit
# ----------------------------

def signal_handler(sig, frame):
	console.stop()
	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ----------------------------
# Game loop
# ----------------------------

while True:
	start = time.time()

	gamestate = console.step()

	consoleStep = time.time()

	if gamestate is None:
		continue

	if console.processingtime * 1000 > 17:
		print("WARNING frame time:", console.processingtime * 1000)

	if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
		raw_frame, action_state, _ = processGamestate(
			gamestate, myPort, opPort
		)

		frame_tensor = torch.tensor(raw_frame, dtype=torch.float32, device=device)

		action_tensor = torch.tensor([action_state], dtype=torch.long, device=device)
		action_embed = embed(action_tensor).squeeze(0)

		lstm_input = frame_tensor.unsqueeze(0).unsqueeze(0)

		with torch.no_grad():
			lstm_out, hidden = lstm(lstm_input, hidden)

		actionAndLSTM = time.time()

		lstm_vec = lstm_out.squeeze()

		hidden = (hidden[0].detach(), hidden[1].detach())

		full_state = torch.cat([
			frame_tensor,
			action_embed,
			lstm_vec
		])

		frame_history.append(full_state)

		if len(frame_history) > CONFIG["history_frames"] + 1:
			frame_history.pop(0)

		if len(input_history) > CONFIG["history_frames"]:
			input_history.pop(0)

		padded_frames = (
			[torch.zeros(full_state_size, device=device)
			for _ in range(CONFIG["history_frames"] + 1 - len(frame_history))]
			+ frame_history
		)

		padded_frames = padded_frames[-(CONFIG["history_frames"] + 1):]

		padded_inputs = (
			[torch.zeros(CONFIG["controller_output_size"], device=device)
			for _ in range(CONFIG["history_frames"] - len(input_history))]
			+ input_history
		)

		padded_inputs = padded_inputs[-CONFIG["history_frames"]:]

		nn_input = torch.cat(padded_frames + padded_inputs)

		preModel = time.time()

		with torch.no_grad():
			pred = model(nn_input)

		postModel = time.time()

		controller_np = pred.detach().cpu().numpy()
		pcs = PickleableControllerState(np_array=controller_np)

		# print(controller_np[csButtonKeys.index(melee.enums.Button.BUTTON_L)])
		# print(controller_np[csButtonKeys.index(melee.enums.Button.BUTTON_R)])

		set_controller_state(
			controller,
			pcs
		)

		input_tensor = torch.tensor(controller_np, dtype=torch.float32, device=device)

		input_history.append(input_tensor)

		end = time.time()

		print("consoleStep: " + str(int(1000 * (consoleStep - start))))
		print("actionAndLSTM: " + str(int(1000 * (actionAndLSTM - consoleStep))))
		print("preModel: " + str(int(1000 * (preModel - actionAndLSTM))))
		print("postModel: " + str(int(1000 * (postModel - preModel))))
		print("end: " + str(int(1000 * (end - postModel))))
	else:

		melee.MenuHelper.menu_helper_simple(
			gamestate,
			controller,
			character,
			stage,
			connect_code,
			costume=costume,
			autostart=True,
			swag=False
		)