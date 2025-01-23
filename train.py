from sklearn import datasets
import torch
from torch import Tensor

from mlp_half_precision import HalfMLP

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(train_x: Tensor, train_y: Tensor, test_x: Tensor, test_y: Tensor,
          epochs: int = 100) -> HalfMLP:
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    best_model = None
    best_dropout = None
    best_accuracy = 0.0

    for dropout in [0.0, 0.1, 0.2, 0.3, 0.4]:
        model = HalfMLP(train_x.shape[1], 128, 1, dropout).to(DEVICE)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters())
        # sgd_optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        print(f"Training model with dropout={dropout}")
        model.train()
        for epoch in range(epochs):
            epoch_losses = []
            # if epoch < 80:
            #     optimizer = adam_optimizer
            # else:
            #     optimizer = sgd_optimizer

            for train_x, train_y in train_loader:
                train_x = train_x.to(DEVICE)
                train_y = train_y.to(DEVICE)
                optimizer.zero_grad()
                output = model(train_x)
                loss = criterion(output, train_y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {sum(epoch_losses) / len(epoch_losses)}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for test_x, test_y in test_loader:
                test_x = test_x.to(DEVICE)
                test_y = test_y.to(DEVICE)
                output = model(test_x)
                predictions = torch.sigmoid(output) > 0.5
                correct += (predictions == test_y.unsqueeze(1)).sum().item()
                total += test_y.shape[0]
        accuracy = correct / total
        print(f"Accuracy: {accuracy}")
        if accuracy > best_accuracy:
            best_model = model
            best_dropout = dropout
            best_accuracy = accuracy
    print(f"Best dropout: {best_dropout}, Best accuracy: {best_accuracy}")
    return best_model

def preprocess(X, y) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # Split the data into training and test sets
    train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(X, y, test_size=0.2,
                                                                                stratify=y, random_state=42)
    # Convert the data to PyTorch tensors
    train_x = torch.tensor(train_x, dtype=torch.float32).to(DEVICE)
    train_y = torch.tensor(train_y, dtype=torch.float32).to(DEVICE)
    test_x = torch.tensor(test_x, dtype=torch.float32).to(DEVICE)
    test_y = torch.tensor(test_y, dtype=torch.float32).to(DEVICE)
    return train_x, train_y, test_x, test_y

def main(epochs: int = 100):
    # Load the data
    X, y = datasets.load_breast_cancer(return_X_y=True)
    # Preprocess the data
    train_x, train_y, test_x, test_y = preprocess(X, y)
    # # Train and find the model
    model = train(train_x, train_y, test_x, test_y, epochs=epochs)

if __name__ == "__main__":
    main(epochs=200)