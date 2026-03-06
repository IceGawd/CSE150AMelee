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
	"frame_feature_size": 64,     # processed frame features
	"action_embed_size": 16,
	"controller_output_size": 16,
	"num_action_states": len(melee.enums.Action), 

	# history
	"history_frames": 10,

	# mlp sizes
	"mlp_hidden": 128,

	# lstm
	"lstm_hidden": 128,
	"lstm_layers": 1,

	# training
	"lr": 1e-3,
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
		hidden_size = output_size * math.pow(2, hidden_layers)

		self.lstm_beginning = nn.Parameter(torch.tensor(lstm.lstm_hidden, dtype=torch.float32, device=device))
		self.lstm = MLP_LSTM_Model(
			input_size,
			CONFIG["mlp_hidden"],
			CONFIG["lstm_hidden"],
			CONFIG["lstm_layers"],
			CONFIG["controller_output_size"]
		).to(device)

		self.action_embedding = action_embedding

		self.intro = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
		)

		self.hidden = [nn.Sequential(nn.Linear(hidden_size >> i, hidden >> (i + 1)), nn.ReLU()), for i in range(hidden_layers + 1)]

	def forward(self, x):
		output = self.net(x)

		for layer in self.hidden:
			output = layer(output)

		return output

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def process_frame(last_frames, last_inputs, current_frame, action_state, action_embedder):
	# convert lists to tensors
	frame_tensor = torch.tensor(current_frame, dtype=torch.float32, device=device)

	action_tensor = torch.tensor([action_state], dtype=torch.long, device=device)

	action_embed = action_embedder(action_tensor).squeeze(0)

	# concatenate frame + action embedding
	nn_input = torch.cat([
		frame_tensor,
		action_embed
	])

	# placeholder target
	# replace with your actual controller data
	target_output = torch.zeros(CONFIG["controller_output_size"], device=device)

	return nn_input, target_output

def getFrameAndActionState():
	frame = [random.random() for _ in range(CONFIG["frame_feature_size"])]
	action_state = random.randint(0, 300)

	controller_output = [random.random() for _ in range(CONFIG["controller_output_size"])]

	return frame, action_state, controller_output

def train():
		# 	seq_inputs = []
		# 	seq_targets = []

		# 	last_frames = []
		# 	last_inputs = []

		# 	for t in range(CONFIG["sequence_length"]):
		# 		frame, action_state, controller = getFrameAndActionState()

		# 		inp, target = process_frame(
		# 			last_frames,
		# 			last_inputs,
		# 			frame,
		# 			action_state,
		# 			action_embedder
		# 		)

		# 		seq_inputs.append(inp)
		# 		seq_targets.append(torch.tensor(controller, device=device))

		# 		last_frames.append(frame)
		# 		last_inputs.append(controller)

		# 		total_frames_trained += 1

		# 	seq_inputs = torch.stack(seq_inputs).unsqueeze(0)
		# 	seq_targets = torch.stack(seq_targets).unsqueeze(0)

		# 	pred, _ = model(seq_inputs)

		# 	loss = loss_fn(pred, seq_targets)

		# 	optimizer.zero_grad()
		# 	loss.backward()
		# 	optimizer.step()

		# print("Epoch:", epoch, "Loss:", loss.item())
		# print("Frames trained on:", total_frames_trained)

def saveNNData(file, data):
	global pickles_dir

	print("DON'T QUIT Saving...")

	# pickles_dir + file + ".pt"

	with open(pickles_dir + file + ".pkl", 'wb') as f:
		pickle.dump(data["last_file"], f)

	print("Saved!")

def addNNData(data, gamestate, myPort, opPort):
	pass

def loadNNData(file):
	global pickles_dir

	print("Loading data...")

	data = {"last_file": None}
	
	action_embedder = ActionEmbedding(CONFIG["num_action_states"], CONFIG["action_embed_size"]).to(device)

	input_size = CONFIG["frame_feature_size"] + CONFIG["action_embed_size"]

	lstm

	print("Total parameters:", count_parameters(model))

	data["optimizer"] = optim.Adam(
		list(model.parameters()) + list(action_embedder.parameters()),
		lr=CONFIG["lr"]
	)

	data["loss_fn"] = nn.MSELoss()

	data["model"] = FeedForwardNet(action_embedder, lstm)

	if os.path.exists(path):
		model.load_state_dict(torch.load(path))


	print("Data Loaded!")
	return data


if __name__ == "__main__":
	loopThrough(addNNData, saveNNData, loadNNData, savefile="ice_god_falco")
