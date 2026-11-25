
import torch
import torch.nn as nn


class MSVStackedLSTM(nn.Module):

    def __init__(
        self,
        input_size=15,
        hidden_size=128,
        num_layers=2,
        output_steps=5,
        dropout=0.2
    ):
        super().__init__()

        self.output_steps = output_steps

        # Stacked LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        # Predict 3D coordinates for each future step
        self.fc = nn.Linear(
            hidden_size,
            output_steps * 3
        )

    def forward(self, x):

        # x shape: (batch_size, 20, 15)
        output, _ = self.lstm(x)

        # Use the final LSTM time step
        final_hidden = output[:, -1, :]

        prediction = self.fc(final_hidden)

        # Output shape: (batch_size, 5, 3)
        prediction = prediction.reshape(
            -1,
            self.output_steps,
            3
        )

        return prediction
