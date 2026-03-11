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
	predict_controller,
	count_parameters
)

device = "cuda" if torch.cuda.is_available() else "cpu"

savefile = "ice_god_fox"

character, stage = character_stage[savefile]
costume = 2
connect_code = ""

# ----------------------------
# Load model
# ----------------------------

data = loadNNData(savefile)

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

		raw_frame, action_state, stage_state, opponent_state, _ = processGamestate(
			gamestate, myPort, opPort
		)

		pcs = predict_controller(
			data,
			raw_frame,
			action_state,
			stage_state,
			opponent_state
		)

		set_controller_state(controller, pcs)
	else:

		melee.MenuHelper.menu_helper_simple(
			gamestate,
			controller,
			character,
			stage,
			connect_code,
			costume=costume,
			autostart=True,
			swag=True
		)