import torchvision
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 30
NUM_CLASSES = 47
EMNIST_MAPPING = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                  'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'r', 't']
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.9,1.1))
])


full_dataset = torchvision.datasets.EMNIST(".", download=True, transform=transform, split="balanced")

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def visualize_predictions(model, mapping, dataset, device, num_correct=3, num_wrong=2):
    model.eval()

    correct_samples = []
    wrong_samples = []

    with torch.no_grad():

        max_search = min(len(dataset), 2000)

        for i in range(max_search):
            if len(correct_samples) >= num_correct and len(wrong_samples) >= num_wrong:
                break

            image, true_label = dataset[i]


            input_image = image.unsqueeze(0).to(device)

            output = model(input_image)
            pred_label = output.argmax(dim=1).item()

            if pred_label == true_label and len(correct_samples) < num_correct:
                correct_samples.append((image, true_label, pred_label))
            elif pred_label != true_label and len(wrong_samples) < num_wrong:
                wrong_samples.append((image, true_label, pred_label))

    all_samples = wrong_samples + correct_samples
    num_samples = len(all_samples)

    if num_samples == 0:
        print("Nie znaleziono żadnych próbek do wyświetlenia. Spróbuj zwiększyć max_search w kodzie.")
        return


    fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 4))


    plt.subplots_adjust(top=0.8)

    if num_samples == 1:
        axes = [axes]


    plt.suptitle('Model Predictions (Red=Wrong, Green=Correct)', y=1.05, fontsize=14)

    for idx, (image, true_label, pred_label) in enumerate(all_samples):
        ax = axes[idx]

        image_to_display = image.squeeze().cpu().numpy().T

        ax.imshow(image_to_display, cmap='gray')
        true_char = mapping[true_label]
        pred_char = mapping[pred_label]

        if true_label == pred_label:
            ax.set_title(f'Prawdziwa: {true_char}\nPredykcja: {pred_char}', color='green')
        else:
            ax.set_title(f'Prawdziwa: {true_char}\nPredykcja: {pred_char}', color='red')

        ax.axis('off')

    plt.show()

class EMNISTClassifier(nn.Module):
    def __init__(self, num_classes=47):
        super(EMNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128*3*3, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EMNISTClassifier(num_classes=NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

print(f"--- Starting Training on {device} for {EPOCHS} epochs ---")

for epoch in range(EPOCHS):
    # ---- Training ----
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    avg_train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train

    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    avg_val_loss = running_val_loss / len(test_loader)
    val_acc = 100 * correct_val / total_val

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}], "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

print("--- Training Finished ---")


epochs = range(1, EPOCHS+1)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.subplot(1,2,2)
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, val_accuracies, label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy over Epochs')
plt.legend()

plt.show()

print("--- Visualizing Model Predictions ---")
visualize_predictions(model,EMNIST_MAPPING ,test_dataset, device)