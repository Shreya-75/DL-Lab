import torch
import torch.nn as nn
import numpy as np

# Set random seed
torch.manual_seed(42)

# XOR Input and Output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Y = np.array([[0], [1], [1], [0]], dtype=np.float32)


class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def train_model(model, X, Y, epochs=10000):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    inputs = torch.tensor(X, dtype=torch.float32)
    targets = torch.tensor(Y, dtype=torch.float32)

    for epoch in range(epochs):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')


# Initialize and train model
model = XORModel()
train_model(model, X, Y)

# Verify predictions
print("\nPredictions:")
for x, y in zip(X, Y):
    input_tensor = torch.tensor(x, dtype=torch.float32)
    pred = model(input_tensor)
    binary_pred = 1 if pred.item() > 0.5 else 0
    print(f"Input: {x}, Target: {y[0]}, Prediction: {binary_pred}")
