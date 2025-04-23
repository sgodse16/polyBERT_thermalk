import numpy as np
np.random.seed(123)
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import torch
torch.manual_seed(456)
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score


device = "cuda" if torch.cuda.is_available else "cpu"
print(device)

############## Dataset with smiles encoded using polyBERT tokenizer ####################################################
class Thermalk_Dataset(Dataset):
    def __init__(self, encoded_input, thermalk, device):
        self.encoded_input = encoded_input.to(device)
        self.thermalk = torch.tensor(thermalk).float().to(device)

    def __len__(self):
        self.len = len(self.thermalk)
        return self.len

    def __getitem__(self, i):
        entry = {"input_ids":self.encoded_input["input_ids"][i], 
                 "token_type_ids":self.encoded_input["token_type_ids"][i], 
                 "attention_mask":self.encoded_input["attention_mask"][i]}
        return entry, self.thermalk[i]

############## Thermal-k Predictor with pretrained Polybert and MLIP for downstream regression ##########################

class ThermalK_Predictor(nn.Module):
    def __init__(self, PretrainedModel):
        super(ThermalK_Predictor, self).__init__()
        self.PretrainedModel = deepcopy(PretrainedModel)
        
        self.Regressor = nn.Sequential(
            nn.BatchNorm1d(self.PretrainedModel.config.hidden_size),
            nn.Linear(self.PretrainedModel.config.hidden_size, self.PretrainedModel.config.hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.PretrainedModel.config.hidden_size),
            nn.Linear(self.PretrainedModel.config.hidden_size, 1)
        )

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, encoded_input):
        outputs = self.PretrainedModel(**encoded_input)
        pooled_output = self.mean_pooling(outputs, encoded_input["attention_mask"])
        output = self.Regressor(pooled_output)
        return output

##############################################################################################################

###### Load CSV Data
data = pd.read_csv("PI1070.csv")
data_clean = data.dropna(subset=["thermal_conductivity", "smiles"])

smiles = data_clean["smiles"]
thermalk = data_clean["thermal_conductivity"]

# Train-Test split (80-20)
ind = [i for i in smiles.index]
for i in range(100):
    np.random.shuffle(ind)
train_ind = ind[:int(0.8*len(ind))]
test_ind = ind[int(0.8*len(ind)):]

train_smiles, train_thermalk = smiles[train_ind].to_list(), thermalk[train_ind].to_list()
test_smiles, test_thermalk = smiles[test_ind].to_list(), thermalk[test_ind].to_list()

scaler = StandardScaler()
train_thermalk = scaler.fit_transform(np.array(train_thermalk).reshape(-1,1))
test_thermalk = scaler.transform(np.array(test_thermalk).reshape(-1,1))


############# Tokenize #############
tokenizer = AutoTokenizer.from_pretrained('kuelumbus/polyBERT')
polyBERT = AutoModel.from_pretrained('kuelumbus/polyBERT')

train_input = tokenizer(train_smiles, padding=True, truncation=True, return_tensors='pt')
test_input = tokenizer(test_smiles, padding=True, truncation=True, return_tensors='pt')

train_data = Thermalk_Dataset(encoded_input=train_input, thermalk=train_thermalk, device=device)
test_data = Thermalk_Dataset(encoded_input=test_input, thermalk=test_thermalk, device=device)

# ######## Training ##########

def train_step(model, train_dataloader, optimizer, scheduler, loss_fn):
    model.train()
    for entry, thermalk in train_dataloader:
        optimizer.zero_grad() 
        pred = model(entry)
        loss = loss_fn(pred.squeeze(), thermalk.squeeze())
        loss.backward()
        optimizer.step()
        scheduler.step()
    return None

def test_step(model, train_dataloader, test_dataloader, loss_fn):
    model.eval()
    with torch.no_grad():
        test_mse = 0
        count = 0
        for entry, thermalk in test_dataloader:
            pred = model(entry)
            count += len(thermalk)
            loss = loss_fn(pred.squeeze(), thermalk.squeeze())
            test_mse += loss.item()
        test_mse = test_mse/count

        train_mse = 0
        count = 0
        for entry, thermalk in train_dataloader:
            pred = model(entry)
            count += len(thermalk)
            loss = loss_fn(pred.squeeze(), thermalk.squeeze())
            train_mse += loss.item()
        train_mse = train_mse/count

    return train_mse, test_mse

num_epochs = 1000
warmup_ratio = 0.05
tolerance = 10
batch_size = 16
steps_per_epoch = len(train_data) // batch_size
training_steps = int(steps_per_epoch*num_epochs)
warmup_steps = int(warmup_ratio*training_steps)
save_path = "polyBERT_downstream_thermalk.pt"

train_dataloader = DataLoader(train_data, batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size, shuffle=False)


best_mse, best_test_mse = 1e6, 1e6
count = 0 # For early stopping criterion

loss_fn = nn.MSELoss()

model = ThermalK_Predictor(polyBERT)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps)

model.to(device)

train_mse, test_mse = [], []

torch.cuda.empty_cache()
for epoch in range(1,num_epochs+1):
    train_step(model, train_dataloader, optimizer, scheduler, loss_fn)
    train_mse_, test_mse_= test_step(model, train_dataloader, test_dataloader, loss_fn)
    print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_mse_}, Test Loss: {test_mse_}")

    if test_mse_ < best_test_mse:
        best_test_mse = test_mse_
        count = 0
    else:
        count += 1

    if test_mse_ < best_mse:
        best_mse = test_mse_
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch}
        torch.save(state, save_path)

    if count >= tolerance:
        print("Early stop")
        break

    train_mse.append(train_mse_)
    test_mse.append(test_mse_)

#########################################################################################################################################

print("------------------- Training Complete -------------------")


## Load The Best Model
new_model = ThermalK_Predictor(polyBERT)
new_model.load_state_dict(torch.load(save_path)["model"])

new_model.to(device)

############# Plot training history
plt.figure()
plt.plot(np.arange(0, len(train_mse), 1), train_mse, label='Train Loss')
plt.plot(np.arange(0, len(test_mse), 1), test_mse, label='Test Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.savefig('training_history.png', dpi=300)

####################################### Check Performance #############################################
new_model.eval()
train_predictions, train_true = [], []
test_predictions, test_true = [], []

with torch.no_grad():
    for entry, thermalk in train_dataloader:
        pred = new_model(entry)
        train_predictions.extend(pred.detach().cpu())
        train_true.extend(thermalk.detach().cpu())

    for entry, thermalk in test_dataloader:
        pred = new_model(entry)
        test_predictions.extend(pred.detach().cpu())
        test_true.extend(thermalk.detach().cpu())

## 
train_true, train_predictions = scaler.inverse_transform(train_true), scaler.inverse_transform(train_predictions)
test_true, test_predictions = scaler.inverse_transform(test_true), scaler.inverse_transform(test_predictions)
train_true = np.array(train_true, dtype=float)
train_predictions = np.array(train_predictions, dtype=float)
test_true = np.array(test_true, dtype=float)
test_predictions = np.array(test_predictions, dtype=float)

train_r2 = r2_score(train_true, train_predictions)
test_r2 = r2_score(test_true, test_predictions)

plt.scatter(train_predictions, train_true, label='Train')
plt.scatter(test_predictions, test_true, label='Test')
plt.plot(np.linspace(0,0.8,10), np.linspace(0,0.8,10), color='black')
plt.xlim(0, 0.8)
plt.ylim(0, 0.8)
plt.legend()
plt.xlabel('Predicted Thermal Conductivity (W/m-K)')
plt.ylabel('True Thermal Conductivity (W/m-K)')
plt.title(f"Train R2: {train_r2:.2f}, Test R2: {test_r2:.2f}")
plt.savefig("parity_plot.png", dpi=300)


##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
############################################ Check on PI1M Dataset ###########################################################


big_data = pd.read_csv('PI1M_v2.csv')

new_smiles = big_data["SMILES"].to_list()
SA_score = big_data["SA Score"].to_list()

new_encodings = tokenizer(new_smiles, padding=True, truncation=True, return_tensors='pt')
eval_data = Thermalk_Dataset(encoded_input=new_encodings, thermalk=SA_score, device=device) # SA score as a place holder for thermal-k
eval_dataloader = DataLoader(eval_data, batch_size, shuffle=False)

new_model.eval()

eval_predictions = []
with torch.no_grad():
    for entry, thermalk in eval_dataloader:
        pred = new_model(entry)
        eval_predictions.extend(pred.detach().cpu())

eval_predictions = scaler.inverse_transform(eval_predictions)

plt.figure()
plt.hist(np.array(eval_predictions, dtype=float), label='PI1M', density=True)
plt.hist(np.array(scaler.inverse_transform(train_thermalk), dtype=float), label='Train', density=True)
plt.hist(np.array(scaler.inverse_transform(test_thermalk), dtype=float), label='Test', density=True)
plt.legend()
plt.xlabel('Thermal Conductivity (W/m-K)')
plt.ylabel('Count')
plt.savefig("Data_Distributions.png", dpi=300)

## Save Predictions
np.savez("PI1M_polybert.npz", smiles=new_smiles, SA_score=SA_score, thermalk=eval_predictions)
