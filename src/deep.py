import jiwer
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.loggers import NeptuneLogger
from dataModule import AudioDataModule, CLASSES
from prep.processor import get_feature_extractor, get_tokenizer, FEATURES
from modules import ctcDecoder
import pandas as pd
from io import StringIO
from neptune.types import File

default_config = {
    "decoder": "greedy",
    "n_class": CLASSES,
    "n_feature": FEATURES,
    "batch_size": 16,
    "lr": 1e-7,
    "n_hidden": 256,
    "n_rnn_layers": 8,
    "dropout": 0.2,
}


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x

class DeepSpeech2ishModel(nn.Module):
    
    def __init__(self, n_cnn_layers=3, n_rnn_layers=5, rnn_dim=512, n_class=CLASSES, n_feats=FEATURES, stride=2, dropout=0.1):
        super(DeepSpeech2ishModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x.unsqueeze(1))
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x
    
class DeepSpeech2ishLightningModule(pl.LightningModule):

    def __init__(self, model: torch.nn.Module):
        super(DeepSpeech2ishLightningModule, self).__init__()
        self.model = model

        # Decoder
        self.ctc_decoder = ctcDecoder.GreedyCTCDecoder(tokenizer=get_tokenizer())
        self.decode_mode = 'greedy'

        self.eval_predictions = []
        self.eval_targets = []

        # Loss function
        self.ctc_loss = nn.CTCLoss(
        blank=self.ctc_decoder.get_blank(),
        )

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        self.val_losses = []
        self.val_wers = []
        self.val_cers = []

    def training_step(self, batch, batch_idx):
        spectrograms, labels, input_lengths, label_lengths = batch 
        
        opt = self.optimizers()
        opt.zero_grad()

        outputs = self.model(spectrograms)
        outputs = F.log_softmax(outputs, dim=2)
        outputs = outputs.transpose(0, 1) # (time, batch, n_class)

        input_lengths = input_lengths // 2
        loss = self.ctc_loss(outputs, labels, input_lengths, label_lengths)
        self.manual_backward(loss)
        opt.step()

        sch = self.lr_schedulers()
        sch.step()

        self.log("train/loss", loss.item(), prog_bar=True, sync_dist=True)


    def decode(self, y_hat, y):
        """
        :param y_hat: The predicted values, shape (T, N, C) where T is TimeSteps, N is the batch size and
                        C is the number of classes.
                        In total, they are N matrices of TxC shape, one for each time step a probability distribution
                        over the classes
        :param y: The true values, shape (N, S) where N is the batch size and S is the length of the word.
        :return: decoded values - (N,) strings
        """
        decoded_y = [self.ctc_decoder.decode(encdoing) for encdoing in y]
        decoded_y_hat = self.ctc_decoder(y_hat)
        return decoded_y_hat, decoded_y

    def validation_step(self, batch, batch_idx):
        spectrograms, labels, input_lengths, label_lengths = batch
        outputs = self.model(spectrograms)

        outputs = F.log_softmax(outputs, dim=2)
        outputs = outputs.transpose(0, 1) # (time, batch, n_class)

        input_lengths = input_lengths // 2
        loss = self.ctc_loss(outputs, labels, input_lengths, label_lengths)
        self.val_losses.append(loss.item())

        decoded_preds, decoded_targets = self.decode(outputs, labels)
        
        self.val_wers.append(jiwer.wer(decoded_targets, decoded_preds))
        self.val_cers.append(jiwer.cer(decoded_targets, decoded_preds))


        self.eval_predictions.extend(decoded_preds)
        self.eval_targets.extend(decoded_targets)

        return loss
    
    def on_validation_epoch_end(self, *_):
        self.log("valid/loss", sum(self.val_losses)/len(self.val_losses))
        self.log("valid/wer", sum(self.val_wers)/len(self.val_wers))
        self.log("valid/cer", sum(self.val_cers)/len(self.val_cers))
        
        df = pd.DataFrame({"predictions": self.eval_predictions, "targets": self.eval_targets})
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        self.logger.experiment[f"data/val_epoch_{self.current_epoch}"].upload(File.from_stream(csv_buffer, extension="csv"))

        self.eval_predictions.clear()
        self.eval_targets.clear()

        self.val_losses.clear()
        self.val_wers.clear()
        self.val_cers.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=5e-4, total_steps=self.trainer.estimated_stepping_batches,
            anneal_strategy='linear'
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
    

def train_func(config=None, logger=None, logger_config=None, dm=None, checkpoints=None, num_epochs=50):
    if config is None:
        config = default_config

    if dm is None:
        dm = AudioDataModule(batch_size=config['batch_size'])

    dm.prepare_data()
    dm.setup("fit")

    if logger is None and logger_config is not None:
        logger = NeptuneLogger(**logger_config)

    model = model_module = DeepSpeech2ishLightningModule(model=DeepSpeech2ishModel())

    # log the hyperparameters and not the api key and project name
    logger.run["parameters"] = config

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=logger,
        max_epochs=num_epochs,
        # gradient_clip_val=0.5,
    )
    trainer.fit(model, datamodule=dm, ckpt_path=checkpoints)

    logger.run.stop()
    return trainer


def main():
    from lightning.pytorch.loggers import NeptuneLogger
    from dotenv import load_dotenv
    import os

    API_TOKEN = os.environ.get("LOGGER_API")
    PROJECT_NAME = 'mrobay/Audio-project'

    logger_config = {
        "api_key": API_TOKEN,
        "project_name": PROJECT_NAME,
        "log_model_checkpoints": False
    }

    neptune_logger = NeptuneLogger(project=PROJECT_NAME, api_key=API_TOKEN, log_model_checkpoints=False)
    train_func(logger=neptune_logger)


if __name__ == '__main__':
    main()
    