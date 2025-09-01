import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy


class AudioTransformerClassifier(pl.LightningModule):
    """
    A PyTorch Lightning Module for audio classification using a Transformer-based architecture.
    This model processes input audio features through a series of transformations,
    including a linear projection, positional encoding, Transformer encoder layers,
    and a final classification layer.
    It supports different pooling strategies to aggregate timeseries.
    """

    def __init__(
        self,
        input_length: int,
        num_input_features: int,
        d_model: int = 128,
        num_classes: int = 10,
        num_encoder_layers: int = 4,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pooling_mode: str = "mean",
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_proj = nn.Linear(
            num_input_features, d_model
        )  # (batch, seq_len, num_features) → (batch, seq_len, d_model)

        # learnable positional embedding: shape (1, seq_len, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, input_length, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # enables input shape (batch, seq_len, d_model)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.pooling_mode = pooling_mode

        self.classifier = nn.Linear(d_model, num_classes)

        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.lr = lr

    def pooling_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a pooling operation to the input tensor along the specified dimension.
        Supported pooling modes:
            - "mean": Computes the mean across dimension 1.
            - "sum": Computes the sum across dimension 1.
            - "max": Computes the maximum value across dimension 1.
        Args:
            x (torch.Tensor): Input tensor to be pooled.
        Returns:
            torch.Tensor: The pooled tensor according to the selected pooling mode.
        Raises:
            ValueError: If an invalid pooling mode is specified.
        """

        if self.pooling_mode == "mean":
            return x.mean(dim=1)
        elif self.pooling_mode == "sum":
            return x.sum(dim=1)
        elif self.pooling_mode == "max":
            return x.max(dim=1).values
        else:
            raise ValueError(f"Invalid pooling_mode: {self.pooling_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the audio classification model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, num_features),
                representing a batch of audio feature sequences.
        Returns:
            torch.Tensor: Output logits tensor of shape (batch, num_classes),
                representing the predicted class scores for each input in the batch.
        The forward pass includes:
            - Projecting input features to model dimension.
            - Adding positional embeddings.
            - Passing through the encoder.
            - Applying a pooling function to aggregate sequence information.
            - Classifying the pooled representation.
        """

        # x shape: (batch, seq_len, num_features)
        x = self.input_proj(x)  # → (batch, seq_len, d_model)

        x = (
            x + self.pos_embedding[:, : x.size(1), :]
        )  # add positional embedding for given sequence

        x = self.encoder(x)  # → (batch, seq_len, d_model)

        x = self.pooling_function(x)  # x → (batch, d_model)

        logits = self.classifier(x)  # → (batch, num_classes)

        return logits

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.
        Args:
            batch (tuple): A tuple containing input data (x) and target labels (y).
            batch_idx (int): Index of the current batch.
        Returns:
            torch.Tensor: The computed loss for the current batch.
        Logs:
            train_loss (float): Training loss for the batch.
            train_acc (float): Training accuracy for the batch.
        """

        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.test_acc(logits, y)
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        """
        Configures and returns the optimizer for training the model.
        Returns:
            torch.optim.Adam: An Adam optimizer initialized with the model's parameters and learning rate.
        """

        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def total_params(self) -> int:
        """
        Calculates the total number of parameters in the model.
        Returns:
            int: The total count of parameters.
        """

        return sum(p.numel() for p in self.parameters())
