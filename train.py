import torch
from mineral_mystic import MineralMystic
from preprocessing import load_dataset, split_data, create_dataloaders
import torch.nn
import torch.optim

config = {
    'lr': 0.0001,
    'epochs': 40,
}

def save_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filename: str):
    '''Saves the model and optimizer state to a file.'''
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filename: str) -> (torch.nn.Module, torch.optim.Optimizer):
    '''Loads the model and optimizer state from a file.'''
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

def print_metrics(epoch: int, epochs: int, train_loss: float, val_loss: float, train_acc: float, val_acc: float):
    '''Prints metrics for the current epoch.'''
    print(f"Epoch {epoch+1}/{epochs}:")
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
    print("-----------------------------")

def fit(epochs: int, model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, 
        loss_criterion: torch.nn.Module, optimizer: torch.optim.Optimizer) -> (list, list, list, list):
    '''Train the model.'''
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        num_correct_train, total_samples_train = 0, 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(images)
            loss    = loss_criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted_labels = outputs.max(1)

            total_samples_train += labels.size(0)
            num_correct_train += predicted_labels.eq(labels).sum().item()

        train_losses.append(running_loss/len(train_loader))
        train_accuracy = 100 * num_correct_train / total_samples_train
        train_accuracies.append(train_accuracy)
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        correct_val, total_val = 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss    = loss_criterion(outputs, labels)

                running_val_loss += loss.item()
                
                _, predicted_labels = outputs.max(1)

                total_val   += labels.size(0)
                correct_val += predicted_labels.eq(labels).sum().item()

        val_losses.append(running_val_loss/len(val_loader))
        val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(val_accuracy)
        
        print_metrics(epoch, epochs, train_losses[-1], val_losses[-1], train_accuracies[-1], val_accuracies[-1])

    return train_losses, val_losses, train_accuracies, val_accuracies


if __name__ == '__main__':
    root_folder = 'temp/mineral_classes'
    dataset     = load_dataset(root_folder)

    x_train, x_val, x_test                = split_data(dataset)
    train_loader, val_loader, test_loader = create_dataloaders(dataset, x_train, x_val, x_test)

    model = MineralMystic()

    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer      = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # Uncomment the next line if you want to load a pre-trained model
    # model, optimizer = load_model(model, optimizer, 'cyrstal_vision_model.pth')
    
    train_losses, val_losses, train_accuracies, val_accuracies = fit(
        config['epochs'], model, train_loader, val_loader, loss_criterion, optimizer
    )
    
    # Save the final model
    save_model(model, optimizer, 'cyrstal_vision_model.pth')

    print(f"Overall Training Accuracy: {sum(train_accuracies)/len(train_accuracies):.2f}%")
    print(f"Overall Validation Accuracy: {sum(val_accuracies)/len(val_accuracies):.2f}%")
