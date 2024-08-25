import torch
import torch.nn as nn
from lightning.pytorch.loggers import NeptuneLogger
import pandas as pd
from io import StringIO
from neptune.types import File
import lightning as pl
from dataModule import AudioDataModule, FEATURES, CLASSES
from modules import bi_rnn, ctcDecoder
from jiwer import wer
from prep.processor import get_feature_extractor, get_tokenizer


default_config = {
    "n_class": CLASSES,
    "n_feature": FEATURES,
    "batch_size": 32,
    "lr": 1e-3,
    "n_hidden": 128,
    "n_rnn_layers": 5,
    "dropout": 0.1,
}


class HebrewASR(pl.LightningModule):
    def __init__(self, config: dict):
        super(HebrewASR, self).__init__()
        self.lr = config['lr']
        self.n_hidden = config['n_hidden']
        self.n_feature  = config['n_feature']
        self.n_rnn_layers = config['n_rnn_layers']
        self.n_class = config['n_class']
        self.dropout = config['dropout']

        self.save_hyperparameters(config)
        
        # Evaluation metrics
        self.eval_loss = []
        self.eval_wer = []
        self.eval_predictions = []
        self.eval_targets = []
        self.test_loss = []
        self.test_wer = []
        self.test_predictions = []
        self.test_targets = []

        # Decoder
        self.ctc_decoder = ctcDecoder.GreedyCTCDecoder(tokenizer=get_tokenizer())

        # Loss function
        self.loss = nn.CTCLoss()

        # Input size:  FxT where F is the number of MFCC features and T is the # of time steps
        # Output size: TxC where T is the number of time steps and C is the number of classes
        self.linear_test = nn.Linear(in_features=self.n_feature, out_features=self.n_class)

        self.bi_rnns = nn.ModuleList()
        for i in range(self.n_rnn_layers):
            if i == 0:
                self.bi_rnns.append(bi_rnn.BiRNN(input_size=self.n_feature, hidden_state_dim=self.n_hidden, rnn_type='gru',
                                                  bidirectional=True, dropout=self.dropout))
            else:
                self.bi_rnns.append(bi_rnn.BiRNN(input_size=self.n_hidden*2, hidden_state_dim=self.n_hidden, rnn_type='gru',
                                              bidirectional=True, dropout=self.dropout))

        self.linear_final = nn.Linear(in_features=self.n_hidden*2, out_features=self.n_class)


    def forward(self, x):
        """
        :param x: (N, F, T) where N is the batch size, T is the number of time steps and F is the number of features
        :return:    The predicted values, shape (T, N, C) where T is TimeSteps, N is the batch size and
                    C is the number of classes.
                    In total, they are N matrices of TxC shape, one for each time step a probability distribution
                    over the classes
        """

        # (N, F, T) 
        for rnn in self.bi_rnns:
            x = rnn(x)

        # (N, H, T)
        x = torch.transpose(x, 1, 2)


        # (N, T, H)
        x = self.linear_final(x)

        # (N, T, C)
        x = nn.functional.log_softmax(x, dim=2)
        x = x.permute(1, 0, 2)

        # (T, N, C)
        return x


    def input_lengths(self, x):
        """
        :param x: The input size (N, F, T)
        """
        input_lengths = []
        for i in range(x.shape[0]):
            mfcc = x[i, :, :]
            mask = torch.all(mfcc == -1.5, axis=0)
            mfcc_length = torch.sum(mask == 0)
            input_lengths.append(mfcc_length)
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)
        return input_lengths


    def calc_loss(self, y_hat, y):
        """
        :param y_hat: The predicted values, shape (T, N, C) where T is TimeSteps, N is the batch size and
                        C is the number of classes.
                        In total, they are N matrices of TxC shape, one for each time step a probability distribution
                        over the classes
        :param y: The true values, shape (N, S) where N is the batch size and S is the length of the word.
        :return: CTC loss.
        """
        batch_size = y_hat.shape[1]
        # label_length is the length of the text label. In our case is the length of the word
        # find where the padding starts
        pad_idx = 4
        un_padded_y = [token[token != pad_idx] for token in y]
        label_length = torch.tensor([len(label) for label in un_padded_y]) * torch.ones(batch_size, dtype=torch.long)

        # The input length is number of acutal time steps
        # run along batch axis
        # input is the time steps
        input_lengths =  torch.full(size=(batch_size,), fill_value=y_hat.shape[0], dtype=torch.long)
        return self.loss(y_hat, y, input_lengths, label_length)
    
    def wer(self, y_hat_str, y_str):
        """
        :param y_hat_str: (N,) strings
        :param y_str: (N,) strings
        :return: Word error rate
        """
        wer_value = wer(y_str, y_hat_str)
        # turn the wer into a tensor
        wer_value = torch.tensor(wer_value)
        return wer_value

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
        decoded_y_hat = []
        for i in range(y_hat.shape[1]):
            logits = y_hat[:, i, :]
            decoded_y_hat.append(self.ctc_decoder(logits))
        return decoded_y_hat, decoded_y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # calculate the loss
        loss = self.calc_loss(y_hat, y)

        # log the loss
        self.log_dict({'train_loss': loss.item()})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.calc_loss(y_hat, y)
        decoded_y_hat, decoded_y = self.decode(y_hat, y)
        wer = self.wer(decoded_y_hat, decoded_y)


        self.eval_loss.append(loss)
        self.eval_wer.append(wer)

        self.eval_predictions.extend(decoded_y_hat)
        self.eval_targets.extend(decoded_y)
        return {"val_loss": loss, "val_wer": wer}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.calc_loss(y_hat, y)
        decoded_y_hat, decoded_y = self.decode(y_hat, y)
        wer = self.wer(y_hat, y)

        self.test_loss.append(loss)
        self.test_wer.append(acc)


        self.test_predictions.extend(decoded_y_hat)
        self.test_targets.extend(decoded_y)
        return {"test_loss": loss, "test_wer": wer}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.eval_loss).mean()
        avg_wer = torch.stack(self.eval_wer).mean()
        self.logger.experiment["training/val_avg_loss"].append(value=avg_loss, step=self.current_epoch)
        self.logger.experiment["training/val_avg_wer"].append(value=avg_wer, step=self.current_epoch)
        df = pd.DataFrame({"predictions": self.eval_predictions, "targets": self.eval_targets})
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        self.logger.experiment[f"data/val_epoch_{self.current_epoch}"].upload(File.from_stream(csv_buffer, extension="csv"))
        self.eval_loss.clear()
        self.eval_wer.clear()
        self.eval_predictions.clear()
        self.eval_targets.clear()

    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_loss).mean()
        avg_wer = torch.stack(self.test_wer).mean()
        self.logger.experiment["training/test_avg_loss"].append(value=avg_loss, step=self.current_epoch)
        self.logger.experiment["training/test_avg_wer"].append(value=avg_wer, step=self.current_epoch)
        df = pd.DataFrame({"predictions": self.test_predictions, "targets": self.test_targets})
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        self.logger.experiment[f"data/testd_l_epoch_{self.current_epoch}"].upload(File.from_stream(csv_buffer, extension="csv"))
        self.test_loss.clear()
        self.test_wer.clear()
        self.test_predictions.clear()
        self.test_targets.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train_func(config=None, logger=None, logger_config=None, num_epochs=10):
    if config is None:
        config = default_config

    dm = AudioDataModule(batch_size=config['batch_size'])
    if logger is None and logger_config is not None:
        logger = NeptuneLogger(**logger_config)


    model = HebrewASR(config)    

    # log the hyperparameters and not the api key and project name
    logger.run["parameters"] = config

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=logger,
        max_epochs=num_epochs,
    )
    trainer.fit(model, datamodule=dm)

    logger.run.stop()
    return trainer


def main():
    train_func()


if __name__ == '__main__':
    main()