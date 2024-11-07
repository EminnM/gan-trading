import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import mplfinance as mpf
import torch.nn.functional as F  

# Set the device to GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = r"C:\Users\TKA\Desktop\ai\data\ohlc\mtf\y_final_1.pt"
data = torch.load(data_path)  

data = data[:, :100, :]  
data = data.to(device)  # Move data to the GPU

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, noise_dim=100, feature_size=4, seq_length=100):
        super(Generator, self).__init__()
        self.seq_length = seq_length
        self.feature_size = feature_size
        
        # Define layers of the generator
        self.fc1 = nn.Linear(noise_dim, 256)  # Input noise to 256 units
        self.fc2 = nn.Linear(256, 512)        # 256 to 512
        self.fc3 = nn.Linear(512, 1024)       # 512 to 1024
        self.fc_out = nn.Linear(1024, seq_length * feature_size)  # 1024 to (100 * 4)

    def forward(self, z):
        # Pass the noise through the network
        x = F.relu(self.fc1(z))  # Shape: (batch_size, 256)
        x = F.relu(self.fc2(x))  # Shape: (batch_size, 512)
        x = F.relu(self.fc3(x))  # Shape: (batch_size, 1024)
        
        # Final layer output
        x = self.fc_out(x)  # Shape: (batch_size, 100 * 4)
        
        # Reshape to (batch_size, seq_length, feature_size) for OHLC data
        x = x.view(z.size(0), self.seq_length, self.feature_size)  # Shape: (batch_size, 100, 4)
        
        return x




class Discriminator(nn.Module):
    def __init__(self, seq_length=100, feature_size=4):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=feature_size, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.gru = nn.GRU(input_size=128, hidden_size=512, num_layers=2, batch_first=True)
        self.fc = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape to (batch, feature_size, seq_length) for Conv1d
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Reshape back to (batch, seq_length, conv_out)
        x, _ = self.gru(x)
        x = x[:, -1, :]  # Use the last GRU output
        x = self.dropout(x)
        x = self.fc(x)
        return torch.sigmoid(x)


# Move models to GPU
generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()

lr = 0.0002
beta1 = 0.5
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

num_epochs = 10000000
batch_size = 512
sample_interval = 1000

for epoch in range(num_epochs):
    # Move the data batch to GPU
    real_data = data[torch.randint(0, data.size(0), (batch_size,))]
    real_data = real_data.to(device)  # Ensure real data is on the GPU

    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    optimizer_d.zero_grad()

    real_output = discriminator(real_data)
    d_loss_real = criterion(real_output, real_labels)

    noise = torch.randn(batch_size, 100).to(device)  # Move noise to GPU
    fake_data = generator(noise)
    fake_output = discriminator(fake_data.detach())
    d_loss_fake = criterion(fake_output, fake_labels)

    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    optimizer_d.step()

    optimizer_g.zero_grad()

    fake_output = discriminator(fake_data)
    g_loss = criterion(fake_output, real_labels) 
    g_loss.backward()
    optimizer_g.step()

    if epoch % (sample_interval // 20) == 0:
        # Print losses
        print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

        # Get discriminator predictions
        real_preds = discriminator(real_data).detach().cpu().numpy()  # Get predictions for real data
        fake_preds = discriminator(fake_data).detach().cpu().numpy()  # Get predictions for fake data

        print(f"Discriminator's prediction for real data: {real_preds.mean()}")
        print(f"Discriminator's prediction for fake data: {fake_preds.mean()}")

        # Plot real data
        def plot_data(data, title="Data"):
            df = pd.DataFrame(data[0].cpu().detach().numpy(), columns=['Open', 'High', 'Low', 'Close'])
            df['Date'] = pd.date_range(start='2024-01-01', periods=100, freq='min')
            df.set_index('Date', inplace=True)

            mpf.plot(df, type='candle', style='charles', title=title, ylabel='Price')

        plot_data(real_data, title='Real Data')
        plot_data(fake_data, title='Fake Data (Generated)')

# Save model states
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")
