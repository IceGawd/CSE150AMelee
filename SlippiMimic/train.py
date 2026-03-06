import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import melee

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Bayes")))
from dataset_collector import *

device = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
	# data
	"frame_feature_size": 57,     # processed frame features
	"action_embed_size": 32,
	"controller_output_size": 18,
	"num_action_states": len(melee.enums.Action),  

	# history
	"history_frames": 10,

	# mlp sizes
	"mlp_hidden": 128,

	# lstm
	"lstm_hidden": 128,
	"lstm_layers": 2,

	# training
	"lr": 1e-4,
	"sequence_length": 1000
}

class ActionEmbedding(nn.Module):
	def __init__(self, num_actions, embed_dim):
		super().__init__()
		self.embedding = nn.Embedding(num_actions, embed_dim)

	def forward(self, action_ids):
		return self.embedding(action_ids)

class MLP_LSTM_Model(nn.Module):
	def __init__(self, input_size, mlp_hidden, lstm_hidden, lstm_layers, output_size):
		super().__init__()

		self.input_size = input_size
		self.mlp_hidden = mlp_hidden
		self.lstm_hidden = lstm_hidden
		self.lstm_layers = lstm_layers
		self.output_size = output_size

		self.mlp = nn.Sequential(
			nn.Linear(input_size, mlp_hidden),
			nn.ReLU(),
			nn.Linear(mlp_hidden, mlp_hidden),
			nn.ReLU()
		)

		self.lstm = nn.LSTM(
			input_size=mlp_hidden,
			hidden_size=lstm_hidden,
			num_layers=lstm_layers,
			batch_first=True
		)

		self.head = nn.Linear(lstm_hidden, output_size)

	def forward(self, x, hidden=None):
		x = self.mlp(x)

		out, hidden = self.lstm(x, hidden)

		out = self.head(out)

		return out, hidden

class FeedForwardNet(nn.Module):
	def __init__(self, input_size, output_size):
		super().__init__()

		hidden_layers = math.ceil(math.log2(input_size / output_size))
		hidden_size = int(output_size * (2 ** hidden_layers))

		layers = []

		layers.append(nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU()
		))

		current = hidden_size
		for i in range(hidden_layers):
			next_size = max(output_size, current // 2)

			layers.append(nn.Sequential(
				nn.Linear(current, next_size),
				nn.ReLU()
			))

			current = next_size

		layers.append(nn.Linear(current, output_size))

		self.layers = nn.ModuleList(layers)

	def forward(self, x):

		out = x

		for layer in self.layers:
			out = layer(out)

		return out

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def processGamestate(gamestate, myPort, opPort):
	"""
	Must return

	raw_frame : list[float]
	action_state : int
	controller : list[float]
	"""

	raw_frame = []
	for port in [myPort, opPort]:
		playerstate = gamestate.players[port]
		raw_frame.append(float(playerstate.action_frame))
		raw_frame.extend(playerstate.ecb_bottom)
		raw_frame.extend(playerstate.ecb_left)
		raw_frame.extend(playerstate.ecb_right)
		raw_frame.extend(playerstate.ecb_top)
		raw_frame.append(float(playerstate.facing))
		raw_frame.append(float(playerstate.hitlag_left))
		raw_frame.append(float(playerstate.hitstun_frames_left))
		raw_frame.append(float(playerstate.invulnerability_left))
		raw_frame.append(float(playerstate.invulnerable))
		raw_frame.append(float(playerstate.jumps_left))
		raw_frame.append(float(playerstate.moonwalkwarning))
		raw_frame.append(float(playerstate.off_stage))
		raw_frame.append(float(playerstate.on_ground))
		raw_frame.append(float(playerstate.percent))
		raw_frame.append(float(playerstate.position.x))
		raw_frame.append(float(playerstate.position.y))
		raw_frame.append(float(playerstate.shield_strength))
		raw_frame.append(float(playerstate.speed_air_x_self))
		raw_frame.append(float(playerstate.speed_ground_x_self))
		raw_frame.append(float(playerstate.speed_x_attack))
		raw_frame.append(float(playerstate.speed_y_attack))
		raw_frame.append(float(playerstate.speed_y_self))
		raw_frame.append(float(playerstate.stock))
	raw_frame.append(gamestate.frame)

	action_state = gamestate.players[myPort].action.value

	controller = PickleableControllerState(gamestate.players[myPort].controller_state).to_numpy()

	return raw_frame, action_state, controller

def train_buffer(data):
	d = data["data"]

	if len(d["buffer_pred"]) == 0:
		return

	preds = torch.stack(d["buffer_pred"])
	targets = torch.stack(d["buffer_y"])

	loss = d["loss_fn"](preds, targets)

	d["optimizer"].zero_grad()
	loss.backward()
	d["optimizer"].step()

	print("Training step loss:", loss.item())

	d["losses"].append(loss.item())
	d["buffer_x"].clear()
	d["buffer_pred"].clear()
	d["buffer_y"].clear()

	for dpfh in range(len(d["prev"]["frame_history"])):
		d["prev"]["frame_history"][dpfh] = d["prev"]["frame_history"][dpfh].detach()

	for dpih in range(len(d["prev"]["input_history"])):
		d["prev"]["input_history"][dpih] = d["prev"]["input_history"][dpih].detach()

def saveNNData(file, data):
	global pickles_dir

	train_buffer(data)

	print("DON'T QUIT Saving...")

	path = pickles_dir + file + "_model.pt"

	torch.save({
		"model": data["data"]["model"].state_dict(),
		"lstm": data["data"]["lstm"].state_dict(),
		"embed": data["data"]["embed"].state_dict(),
		"optimizer": data["data"]["optimizer"].state_dict(),
		"frames": data["data"]["frames"], 
		"losses": data["data"]["losses"]
	}, path)

	with open(pickles_dir + file + ".pkl", "wb") as f:
		pickle.dump(data["last_file"], f)

	print("Saved!")

def addNNData(data, gamestate, myPort, opPort):
	raw_frame, action_state, controller = processGamestate(
		gamestate, myPort, opPort
	)

	d = data["data"]

	if "frame_history" not in d["prev"]:
		d["prev"]["frame_history"] = []
		d["prev"]["input_history"] = []
		d["prev"]["hidden"] = None

	frame_hist = d["prev"]["frame_history"]
	input_hist = d["prev"]["input_history"]
	hidden = d["prev"]["hidden"]

	frame_tensor = torch.tensor(raw_frame, dtype=torch.float32, device=device)
	action_tensor = torch.tensor([action_state], dtype=torch.long, device=device)
	action_embed = d["embed"](action_tensor).squeeze(0)
	lstm_input = frame_tensor.unsqueeze(0).unsqueeze(0)
	lstm_out, hidden = d["lstm"](lstm_input, hidden)
	lstm_vec = lstm_out.squeeze().detach()

	hidden = (hidden[0].detach(), hidden[1].detach())

	full_state = torch.cat([
		frame_tensor,
		action_embed,
		lstm_vec
	])

	frame_hist.append(full_state)

	if len(frame_hist) > CONFIG["history_frames"] + 1:
		frame_hist.pop(0)

	if len(input_hist) > CONFIG["history_frames"]:
		input_hist.pop(0)

	full_state_size = (
		CONFIG["frame_feature_size"]
		+ CONFIG["action_embed_size"]
		+ CONFIG["lstm_hidden"]
	)

	# build padded histories
	padded_frames = [torch.zeros(full_state_size, device=device) for _ in range(CONFIG["history_frames"] + 1 - len(frame_hist))] + frame_hist
	padded_frames = padded_frames[-(CONFIG["history_frames"] + 1):]

	padded_inputs = [torch.zeros(CONFIG["controller_output_size"], device=device) for _ in range(CONFIG["history_frames"] - len(input_hist))] + input_hist
	padded_inputs = padded_inputs[-CONFIG["history_frames"]:]

	nn_input = torch.cat(padded_frames + padded_inputs)

	controller_tensor = torch.tensor(
		controller,
		dtype=torch.float32,
		device=device
	)

	input_hist.append(controller_tensor)
	pred = d["model"](nn_input)

	d["buffer_x"].append(nn_input)
	d["buffer_pred"].append(pred)
	d["buffer_y"].append(controller_tensor)

	d["frames"] += 1
	d["prev"]["hidden"] = hidden

	if len(d["buffer_pred"]) >= CONFIG["sequence_length"]:   # 4 seconds of gameplay
		train_buffer(data)

def loadNNData(file):
	global pickles_dir

	print("Loading data...")

	data = {"data": {}, "last_file": None}

	action_embedder = ActionEmbedding(
		CONFIG["num_action_states"],
		CONFIG["action_embed_size"]
	).to(device)

	lstm = MLP_LSTM_Model(
		CONFIG["frame_feature_size"],
		CONFIG["mlp_hidden"],
		CONFIG["lstm_hidden"],
		CONFIG["lstm_layers"],
		CONFIG["lstm_hidden"]
	).to(device)

	full_state_size = (
		CONFIG["frame_feature_size"]
		+ CONFIG["action_embed_size"]
		+ CONFIG["lstm_hidden"]
	)

	ffn_input_size = (
		(CONFIG["history_frames"] + 1) * full_state_size
		+ CONFIG["history_frames"] * CONFIG["controller_output_size"]
	)

	model = FeedForwardNet(
		ffn_input_size,
		CONFIG["controller_output_size"]
	).to(device)

	optimizer = optim.Adam(
		list(model.parameters())
		+ list(lstm.parameters())
		+ list(action_embedder.parameters()),
		lr=CONFIG["lr"]
	)

	data["data"]["model"] = model
	data["data"]["lstm"] = lstm
	data["data"]["embed"] = action_embedder
	data["data"]["optimizer"] = optimizer
	data["data"]["loss_fn"] = nn.MSELoss()
	data["data"]["frames"] = 0
	data["data"]["losses"] = []
	data["data"]["buffer_x"] = []
	data["data"]["buffer_pred"] = []
	data["data"]["buffer_y"] = []
	data["data"]["prev"] = {}

	path = pickles_dir + file + "_model.pt"

	if os.path.exists(path):

		checkpoint = torch.load(path, map_location=device)

		model.load_state_dict(checkpoint["model"])
		lstm.load_state_dict(checkpoint["lstm"])
		action_embedder.load_state_dict(checkpoint["embed"])
		optimizer.load_state_dict(checkpoint["optimizer"])
		data["data"]["frames"] = checkpoint["frames"]
		data["data"]["losses"] = checkpoint["losses"]

	last_file_path = pickles_dir + file + ".pkl"

	if os.path.exists(last_file_path):
		with open(last_file_path, "rb") as f:
			data["last_file"] = pickle.load(f)

	print("Total parameters:", count_parameters(model) + count_parameters(lstm) + count_parameters(action_embedder))

	print("Data Loaded!")

	return data


if __name__ == "__main__":
	loopThrough(addNNData, saveNNData, loadNNData, savefile="ice_god_samus")
