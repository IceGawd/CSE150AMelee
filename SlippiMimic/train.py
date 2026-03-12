import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import melee
import datetime
import math

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Bayes")))
from dataset_collector import *

device = "cuda" if torch.cuda.is_available() else "cpu"

action_list = list(melee.enums.Action)
stage_list = list(melee.enums.Stage)
opponent_list = list(melee.enums.Action)

CONFIG = {
	# data
	"frame_feature_size": 35,     # processed frame features
	"controller_output_size": 18,

	# embeds
	"action_embed_size": 16,
	"stage_embed_size": 16,
	"opponent_embed_size": 16,
	"num_action_states": len(action_list), 
	"num_stage_states": len(stage_list), 
	"num_opponent_states": len(opponent_list), 

	# ffn
	"ffn_hidden": 128, 
	"ffn_layers": 3, 

	# history
	"history_frames": 4,

	# mlp sizes
	"mlp_hidden": 32,

	# lstm
	"lstm_hidden": 32,
	"lstm_layers": 2,

	# training
	"lr": 1e-3, 
	"sequence_length": 100, 
	"EPOCHS": 10
}

quitAfterThis = False

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
	def __init__(self, input_size, output_size, hidden_size=256, layers=3):
		super().__init__()

		net = []
		factor = int(max(1, math.ceil(math.pow(4 * input_size / hidden_size, 1.0 / layers))))

		in_size = input_size
		out_size = int(math.pow(factor, layers) * hidden_size)

		for _ in range(layers):
			net.append(nn.Linear(in_size, out_size))
			net.append(nn.ReLU())

			in_size = out_size
			out_size //= factor

		net.append(nn.Linear(in_size, output_size))

		self.net = nn.Sequential(*net)

	def forward(self, x):
		return self.net(x)

class MixedControllerLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.button_count = len(csButtonKeys)
		self.bcell = nn.BCEWithLogitsLoss()
		self.bce = nn.BCELoss()
		self.mse = nn.MSELoss()
		self.sigmoid = nn.Sigmoid()

	def magnitude(self, stick):
		return (torch.pow(2 * stick[:, 0] - 1, 2) + torch.pow(2 * stick[:, 1] - 1, 2)).unsqueeze(1)

	def angle(self, stick):
		return torch.atan2(stick[:, 1] - 0.5, stick[:, 0] - 0.5).unsqueeze(1)

	def angleDiff(self, stick_pred, stick_target):
		angle_pred = self.angle(stick_pred)
		angle_target = self.angle(stick_target)

		diff = angle_pred - angle_target

		# wrap to [-pi, pi]
		diff = torch.atan2(torch.sin(diff), torch.cos(diff))

		# normalize to [0,1]
		return torch.abs(diff) / math.pi

	def forward(self, pred, target):
		# Layout assumption
		# [buttons][c_x][c_y][l][main_x][main_y][r]

		buttons_pred = pred[:, :self.button_count]
		buttons_target = target[:, :self.button_count]

		idx = self.button_count

		c_stick_pred = pred[:, idx:idx+2]
		c_stick_target = target[:, idx:idx+2]
		idx += 2

		l_shoulder_pred = pred[:, idx:idx+1]
		l_shoulder_target = target[:, idx:idx+1]
		idx += 1

		main_stick_pred = pred[:, idx:idx+2]
		main_stick_target = target[:, idx:idx+2]
		idx += 2

		r_shoulder_pred = pred[:, idx:idx+1]
		r_shoulder_target = target[:, idx:idx+1]

		# ---- BCE for everything except sticks ----
		logits_pred = torch.cat([
			buttons_pred,
			l_shoulder_pred,
			r_shoulder_pred
		], dim=1)

		logits_target = torch.cat([
			buttons_target,
			l_shoulder_target,
			r_shoulder_target
		], dim=1)

		# print(logits_pred.shape)
		# print(logits_target.shape)

		# c_stick_pred = self.sigmoid(c_stick_pred)
		# main_stick_pred = self.sigmoid(main_stick_pred)

		analog_pred = torch.cat([
			self.magnitude(c_stick_pred),
			self.angleDiff(c_stick_pred, c_stick_target), 
			self.magnitude(main_stick_pred), 
			self.angleDiff(main_stick_pred, main_stick_target)
		], dim=1)

		smcst = self.magnitude(c_stick_target)
		smmst = self.magnitude(main_stick_target)

		analog_target = torch.cat([
			smcst,
			torch.zeros_like(smcst), 
			smmst,
			torch.zeros_like(smmst)
		], dim=1)

		# analog_pred = torch.cat([
		# 	c_stick_pred,
		# 	main_stick_pred
		# ], dim=1)

		# analog_target = torch.cat([
		# 	c_stick_target,
		# 	main_stick_target
		# ], dim=1)

		# print(torch.min(analog_pred))
		# print(torch.max(analog_pred))	
		# print(torch.min(analog_target))
		# print(torch.max(analog_target))

		print(main_stick_pred[0, :])
		print(main_stick_target[0, :])
		# print(self.magnitude(main_stick_pred[0:1, :]))
		# print(self.angle(main_stick_pred[0:1, :]))
		# print(self.magnitude(main_stick_target[0:1, :]))
		# print(self.angle(main_stick_target[0:1, :]))

		loss_logits = self.bcell(logits_pred, logits_target)
		loss_analog = self.mse(analog_pred, analog_target)

		print(loss_logits)

		return loss_logits

		# return self.bcell(pred, target)

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def processGamestate(gamestate, myPort, opPort):
	"""
	Must return

	raw_frame : list[float]
	action_state : int
	stage_state : int
	opponent_state : int
	controller : list[float]
	"""

	raw_frame = []
	for port in [myPort, opPort]:
		playerstate = gamestate.players[port]
		raw_frame.append(float(playerstate.action_frame))
		# raw_frame.extend(playerstate.ecb_bottom)
		# raw_frame.extend(playerstate.ecb_left)
		# raw_frame.extend(playerstate.ecb_right)
		# raw_frame.extend(playerstate.ecb_top)
		raw_frame.append(float(playerstate.facing))
		raw_frame.append(float(playerstate.hitlag_left))
		raw_frame.append(float(playerstate.hitstun_frames_left))
		raw_frame.append(float(playerstate.invulnerability_left))
		raw_frame.append(float(playerstate.invulnerable))
		raw_frame.append(float(playerstate.jumps_left))
		# raw_frame.append(float(playerstate.moonwalkwarning))
		# raw_frame.append(float(playerstate.off_stage))
		# raw_frame.append(float(playerstate.on_ground))
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

	action_state = action_list.index(gamestate.players[myPort].action)
	stage_state = stage_list.index(gamestate.stage)
	opponent_state = opponent_list.index(gamestate.players[opPort].action)

	pcs = PickleableControllerState(gamestate.players[myPort].controller_state)
	controller = pcs.to_numpy()
	# print(pcs.main_stick)

	return raw_frame, action_state, stage_state, opponent_state, controller

def train_buffer(data):
	d = data["data"]

	for i in range(100):
		for (raw_frame, action_state, stage_state, opponent_state, controller) in d["raw_data"]:
			predictFromGamestate(data, raw_frame, action_state, stage_state, opponent_state, controller)

		if len(d["buffer_pred"]) == 0:
			return

		preds = torch.stack(d["buffer_pred"])
		targets = torch.stack(d["buffer_y"])

		loss = d["loss_fn"](preds, targets)

		d["optimizer"].zero_grad()
		loss.backward()
		d["optimizer"].step()

		print(i)
		print("Training step loss:", loss.item())

		for dpfh in range(len(d["prev"]["frame_history"])):
			d["prev"]["frame_history"][dpfh] = d["prev"]["frame_history"][dpfh].detach()

		for dpih in range(len(d["prev"]["input_history"])):
			d["prev"]["input_history"][dpih] = d["prev"]["input_history"][dpih].detach()

		d["losses"].append(loss.item())
		d["buffer_pred"].clear()
		d["buffer_y"].clear()

	d["raw_data"].clear()


def saveNNData(file, data):
	global pickles_dir

	train_buffer(data)

	print("DON'T QUIT Saving...")

	path = pickles_dir + file + "_model.pt"

	torch.save({
		"model": data["data"]["model"].state_dict(),
		"lstm": data["data"]["lstm"].state_dict(),
		"action": data["data"]["action"].state_dict(),
		"stage": data["data"]["stage"].state_dict(),
		"opponent": data["data"]["opponent"].state_dict(),
		"optimizer": data["data"]["optimizer"].state_dict(),
		"frames": data["data"]["frames"], 
		"losses": data["data"]["losses"], 
		"epochs": data["data"]["epochs"], 
		"config": CONFIG
	}, path)

	with open(pickles_dir + file + ".pkl", "wb") as f:
		pickle.dump(data["last_file"], f)

	print("Saved!")

def addNNData(data, gamestate, myPort, opPort):
	raw_frame, action_state, stage_state, opponent_state, controller = processGamestate(
		gamestate, myPort, opPort
	)

	# predictFromGamestate(data, raw_frame, action_state, stage_state, opponent_state, controller)

	data["data"]["raw_data"].append((raw_frame, action_state, stage_state, opponent_state, controller))

	if len(data["data"]["raw_data"]) >= CONFIG["sequence_length"]:
		train_buffer(data)

def predict_controller(data, raw_frame, action_state, stage_state, opponent_state, training=False):
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
	stage_tensor = torch.tensor([stage_state], dtype=torch.long, device=device)
	opponent_tensor = torch.tensor([opponent_state], dtype=torch.long, device=device)

	if training:
		action_embed = d["action"](action_tensor).squeeze(0)
		stage_embed = d["stage"](stage_tensor).squeeze(0)
		opponent_embed = d["opponent"](opponent_tensor).squeeze(0)

		lstm_input = frame_tensor.unsqueeze(0).unsqueeze(0)
		lstm_out, hidden = d["lstm"](lstm_input, hidden)
	else:
		with torch.no_grad():
			action_embed = d["action"](action_tensor).squeeze(0)
			stage_embed = d["stage"](stage_tensor).squeeze(0)
			opponent_embed = d["opponent"](opponent_tensor).squeeze(0)

			lstm_input = frame_tensor.unsqueeze(0).unsqueeze(0)
			lstm_out, hidden = d["lstm"](lstm_input, hidden)

	lstm_vec = lstm_out.squeeze()

	# detach hidden state so BPTT doesn't explode
	hidden = (hidden[0].detach(), hidden[1].detach())

	full_state = torch.cat([
		frame_tensor,
		action_embed,
		stage_embed,
		opponent_embed,
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
		+ CONFIG["stage_embed_size"]
		+ CONFIG["opponent_embed_size"]
		+ CONFIG["lstm_hidden"]
	)

	padded_frames = [torch.zeros(full_state_size, device=device)
		for _ in range(CONFIG["history_frames"] + 1 - len(frame_hist))] + frame_hist
	padded_frames = padded_frames[-(CONFIG["history_frames"] + 1):]

	padded_inputs = [torch.zeros(CONFIG["controller_output_size"], device=device)
		for _ in range(CONFIG["history_frames"] - len(input_hist))] + input_hist
	padded_inputs = padded_inputs[-CONFIG["history_frames"]:]

	nn_input = torch.cat(padded_frames + padded_inputs)

	if training:
		pred = d["model"](nn_input)
	else:
		with torch.no_grad():
			pred = d["model"](nn_input)

	d["prev"]["hidden"] = hidden

	if training:
		return pred          # keep gradients
	else:
		pred_np = pred.detach().cpu().numpy()

		pcs = PickleableControllerState(np_array=pred_np)

		input_hist.append(torch.tensor(pcs.to_numpy(), dtype=torch.float32, device=device)) # do PickleableController sampling change in future
		return pcs

def predictFromGamestate(data, raw_frame, action_state, stage_state, opponent_state, controller):
	d = data["data"]

	pred = predict_controller(
		data,
		raw_frame,
		action_state,
		stage_state,
		opponent_state,
		training=True
	)

	controller_tensor = torch.tensor(controller, dtype=torch.float32, device=device)

	# teacher forcing: real controller goes into history
	d["prev"]["input_history"].append(controller_tensor)

	d["buffer_pred"].append(pred)
	d["buffer_y"].append(controller_tensor)

	d["frames"] += 1

def loadNNData(file):
	global pickles_dir
	global quitAfterThis
	global CONFIG

	print("Loading data...")

	data = {"data": {}, "last_file": None}

	path = pickles_dir + file + "_model.pt"
	checkpoint = None

	if os.path.exists(path):
		checkpoint = torch.load(path, map_location=device)
		if "config" in checkpoint:
			CONFIG = checkpoint["config"]

	action_embedder = nn.Embedding(
		CONFIG["num_action_states"],
		CONFIG["action_embed_size"]
	).to(device)

	stage_embedder = nn.Embedding(
		CONFIG["num_stage_states"],
		CONFIG["stage_embed_size"]
	).to(device)

	opponent_embedder = nn.Embedding(
		CONFIG["num_opponent_states"],
		CONFIG["opponent_embed_size"]
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
		+ CONFIG["stage_embed_size"]
		+ CONFIG["opponent_embed_size"]
		+ CONFIG["lstm_hidden"]
	)

	ffn_input_size = (
		(CONFIG["history_frames"] + 1) * full_state_size
		+ CONFIG["history_frames"] * CONFIG["controller_output_size"]
	)

	model = FeedForwardNet(
		ffn_input_size,
		CONFIG["controller_output_size"], 
		CONFIG["ffn_hidden"], 
		CONFIG["ffn_layers"]
	).to(device)

	optimizer = optim.Adam(
		list(model.parameters())
		+ list(lstm.parameters())
		+ list(action_embedder.parameters())
		+ list(stage_embedder.parameters())
		+ list(opponent_embedder.parameters()), 
		lr=CONFIG["lr"]
	)

	data["data"]["model"] = model
	data["data"]["lstm"] = lstm
	data["data"]["action"] = action_embedder
	data["data"]["stage"] = stage_embedder
	data["data"]["opponent"] = opponent_embedder
	data["data"]["optimizer"] = optimizer
	data["data"]["loss_fn"] = MixedControllerLoss()
	data["data"]["frames"] = 0
	data["data"]["losses"] = []
	data["data"]["epochs"] = -1
	data["data"]["buffer_pred"] = []
	data["data"]["buffer_y"] = []
	data["data"]["raw_data"] = []
	data["data"]["prev"] = {}

	if checkpoint:
		model.load_state_dict(checkpoint["model"])
		lstm.load_state_dict(checkpoint["lstm"])
		action_embedder.load_state_dict(checkpoint["action"])
		stage_embedder.load_state_dict(checkpoint["stage"])
		opponent_embedder.load_state_dict(checkpoint["opponent"])
		optimizer.load_state_dict(checkpoint["optimizer"])
		data["data"]["frames"] = checkpoint["frames"]
		data["data"]["losses"] = checkpoint["losses"]

		if "epochs" in checkpoint:
			data["data"]["epochs"] = checkpoint["epochs"]
		else:
			data["data"]["epochs"] = 0

	last_file_path = pickles_dir + file + ".pkl"

	if os.path.exists(last_file_path):
		with open(last_file_path, "rb") as f:
			data["last_file"] = pickle.load(f)
	else:
		data["data"]["epochs"] += 1

	if data["data"]["epochs"] >= CONFIG["EPOCHS"] - 1:
		quitAfterThis = True

	print("EPOCHS:", data["data"]["epochs"])

	print("CONFIG:")
	for k,v in CONFIG.items():
		print(k, ":", v)

	print("Total parameters:", count_parameters(model)
		+ count_parameters(lstm)
		+ count_parameters(action_embedder)
		+ count_parameters(stage_embedder)
		+ count_parameters(opponent_embedder)
	)

	print("Frames trained on:", data["data"]["frames"])

	print("Data Loaded!")

	return data


if __name__ == "__main__":
	savefile = "ice_god_fox"

	while not quitAfterThis:
		loopThrough(addNNData, saveNNData, loadNNData, savefile=savefile)

		# reset dataset progress for next epoch
		last_file_path = pickles_dir + savefile + ".pkl"
		if os.path.exists(last_file_path):
			os.remove(last_file_path)

		print(datetime.datetime.now())