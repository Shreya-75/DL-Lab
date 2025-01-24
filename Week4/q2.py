import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the input and output for XOR
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Define the XOR model with ReLU
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.linear1 = nn.Linear(2, 2, bias=True)
        self.activation1 = nn.ReLU()  # Changed from Sigmoid to ReLU
        self.linear2 = nn.Linear(2, 1, bias=True)
        self.activation2 = nn.Sigmoid()  # Kept Sigmoid for output layer

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        return x

# Custom Dataset remains the same
class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Create dataset and data loader
full_dataset = MyDataset(X, Y)
batch_size = 1
train_data_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = XORModel().to(device)
print(model)

# Loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Training function
def train_one_epoch():
    total_loss = 0
    for data in train_data_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_data_loader)

# Training loop
epochs = 5000
loss_list = []
for epoch in range(epochs):
    model.train()
    avg_loss = train_one_epoch()
    loss_list.append(avg_loss)

# Print model parameters
print("Model Parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")

# Inference
model.eval()
input_data = torch.tensor([0, 1], dtype=torch.float32).to(device)
with torch.no_grad():
    output = model(input_data.unsqueeze(0))
    output = output.cpu().numpy()
    output = 1 if output > 0.5 else 0
    print("Inputs = {}".format(input_data.cpu().numpy()))
    print("Output y predicted = {}".format(output))

# Plot loss
plt.plot(loss_list)
plt.title('Training Loss with ReLU')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
