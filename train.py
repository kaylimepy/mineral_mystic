import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import CrystalVision
from preprocessing import get_processed_data
import numpy

def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001):
    """
    Train the CrystalVision model.

    Parameters:
    - model (nn.Module): The neural network model to train.
    - train_loader, val_loader (DataLoader): Data loaders for training and validation sets.
    - num_epochs (int): Number of training epochs.
    - lr (float): Learning rate for the optimizer.

    Returns:
    - model (nn.Module): Trained neural network model.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss    = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        # Evaluation on the validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return model

def save_model(model, path='crystal_vision.pth'):
    """
    Save the CrystalVision model to a file.

    Parameters:
    - model (nn.Module): The neural network model to save.
    - path (str): Path where the model should be saved.
    """
    torch.save(model.state_dict(), path)

def load_model(path='crystal_vision.pth'):
    """
    Load the CrystalVision model from a file.

    Parameters:
    - path (str): Path from where the model should be loaded.

    Returns:
    - model (nn.Module): Loaded neural network model.
    """
    model = CrystalVision()
    model.load_state_dict(torch.load(path))
    return model

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_processed_data()

    # Convert the lists to TensorDataset
    train_data = TensorDataset(torch.tensor(numpy.array(X_train)).float().permute(0, 3, 1, 2) / 255.0, torch.tensor(y_train))
    val_data   = TensorDataset(torch.tensor(numpy.array(X_val)).float().permute(0, 3, 1, 2) / 255.0, torch.tensor(y_val))
    test_data  = TensorDataset(torch.tensor(numpy.array(X_test)).float().permute(0, 3, 1, 2) / 255.0, torch.tensor(y_test))

    # DataLoader instantiation
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=32)

    model         = CrystalVision()
    trained_model = train_model(model, train_loader, val_loader)

    # Save the trained model
    save_model(trained_model, 'crystal_vision.pth')
