import torch
import torch.nn as nn
from lightning.pytorch.loggers import NeptuneLogger
import pandas as pd
from io import StringIO
from neptune.types import File
import lightning as pl
from dataModule import AudioDataModule, CLASSES
from modules import bi_rnn, ctcDecoder, convolution, danse
from modules.danse import LayerNorm, Danse
from jiwer import wer
from prep.processor import get_feature_extractor, get_tokenizer, FEATURES


default_config = {
    "decoder": "greedy",
    "n_class": CLASSES,
    "n_feature": FEATURES,
    "batch_size": 16,
    "lr": 9e-5,
    "n_hidden": 256,
    "n_rnn_layers": 8,
    "dropout": 0.2,
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
        if config['decoder'] == 'greedy':
            self.ctc_decoder = ctcDecoder.GreedyCTCDecoder(tokenizer=get_tokenizer())
        elif config['decoder'] == 'beam':
            self.ctc_decoder = ctcDecoder.BeamCTCDecoder(tokenizer=get_tokenizer())
        self.decode_mode = config['decoder']



        # Loss function
        self.loss = nn.CTCLoss(
        zero_infinity=True,
        blank=self.ctc_decoder.get_blank(),
        )

        # Input size:  FxT where F is the number of MFCC features and T is the # of time steps
        # Output size: TxC where T is the number of time steps and C is the number of classes
    
        self.conv_layers = convolution.CNN(input_dim=self.n_feature)

        self.bi_rnns = nn.ModuleList()
        rnn_output_size = self.n_hidden * 2
        for i in range(self.n_rnn_layers):
            if i == 0:
                self.bi_rnns.append(bi_rnn.BiRNN(input_size=self.conv_layers.get_output_dim(), 
                                                hidden_state_dim=self.n_hidden, 
                                                rnn_type='gru',
                                                bidirectional=True,
                                                dropout=self.dropout))
            else:
                self.bi_rnns.append(bi_rnn.BiRNN(input_size=rnn_output_size, 
                                                hidden_state_dim=self.n_hidden, 
                                                rnn_type='gru',
                                                bidirectional=True, 
                                                dropout=self.dropout))


        self.linear_final = nn.Sequential(
            LayerNorm(rnn_output_size),
            Danse(rnn_output_size, rnn_output_size, bias=True),
            nn.ReLU(),
            LayerNorm(rnn_output_size),
            Danse(rnn_output_size, self.n_class, bias=False)
        )


    def forward(self, x):
        """
        :param x: (N, F, T) where N is the batch size, T is the number of time steps and F is the number of features
        :return:  The predicted values, shape (T, N, C) where T is TimeSteps, N is the batch size and
                C is the number of classes.
                In total, they are N matrices of TxC shape, one for each time step a probability distribution
                over the classes
        """
        # (N, F, T) 
        x = self.conv_layers(x)

        # (N, H, T/2) 
        for rnn in self.bi_rnns:
            x = rnn(x)

        # (N, H, T/2) 
        x = torch.transpose(x, 1, 2)


        # (N, T/2, H)
        x = self.linear_final(x)

        # (N, T/2, C)
        x = nn.functional.log_softmax(x, dim=2)

        # (N, T/2, C)
        x = torch.transpose(x, 0, 1)

        # (T/2, N, C)
        return x


    def calc_loss(self, y_hat, y, y_hat_len, y_len):
        """
        :param y_hat: The predicted values, shape (T, N, C) where T is TimeSteps, N is the batch size and
                        C is the number of classes.
                        In total, they are N matrices of TxC shape, one for each time step a probability distribution
                        over the classes
        :param y: The true values, shape (N, S) where N is the batch size and S is the length of the word.
        :param y_hat_len: The length of the predicted values, shape (N,)
        :param y_len: The length of the true values, shape (N,)
        :return: CTC loss.
        """
        batch_size = y_hat.shape[1]
        # The input length is number of acutal time steps
        # run along batch axis
        # input is the time steps
        input_lengths = torch.full(size=(batch_size,), fill_value=y_hat.shape[0], dtype=torch.long)

        return self.loss(y_hat, y, input_lengths, y_len)
    
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
        decoded_y_hat = self.ctc_decoder(y_hat)
        return decoded_y_hat, decoded_y

    def training_step(self, batch, batch_idx):
        x, x_len, y, y_len = batch
        y_hat = self(x)
        # calculate the loss
        loss = self.calc_loss(y_hat, y, x_len, y_len)

        # log the loss
        self.log_dict({'train_loss': loss.item()}, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_len, y, y_len = batch
        y_hat = self(x)
        loss = self.calc_loss(y_hat, y, x_len, y_len)
        decoded_y_hat, decoded_y = self.decode(y_hat, y)
        wer = self.wer(decoded_y_hat, decoded_y)

        self.eval_loss.append(loss)
        self.eval_wer.append(wer)

        self.eval_predictions.extend(decoded_y_hat)
        self.eval_targets.extend(decoded_y)

        return {"val_loss": loss, "val_wer": wer}

    def test_step(self, batch, batch_idx):
        x, x_len, y, y_len = batch

        y_hat = self(x)

        loss = self.calc_loss(y_hat, y, x_len, y_len)

        decoded_y_hat, decoded_y = self.decode(y_hat, y)
        wer = self.wer(decoded_y_hat, decoded_y)

        self.test_loss.append(loss)
        self.test_wer.append(wer)

        self.test_predictions.extend(decoded_y_hat)
        self.test_targets.extend(decoded_y)

        self.log_dict({"test_loss": loss, "test_wer": wer}, on_step=True, on_epoch=False)

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
        self.logger.experiment[f"data/test_epoch_{self.current_epoch}"].upload(File.from_stream(csv_buffer, extension="csv"))
        self.test_loss.clear()
        self.test_wer.clear()
        self.test_predictions.clear()
        self.test_targets.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


    def transcribe(self, audio_bytes):
        """
        :param audio_bytes: The audio file in bytes
        :return: The transcribed text
        """
        feature_extractor = get_feature_extractor()
        tokenizer = get_tokenizer()
        x = feature_extractor(audio_bytes, sampling_rate=16000).input_features[0].T
        x = torch.tensor(x).unsqueeze(0)
        y_hat = self(x)
        decoded_y_hat = self.ctc_decoder(y_hat)
        decoded_y_hat = decoded_y_hat[0]
        return decoded_y_hat


def train_func(config=None, logger=None, logger_config=None, dm=None, checkpoints=None, num_epochs=50):
    if config is None:
        config = default_config

    if dm is None:
        dm = AudioDataModule(batch_size=config['batch_size'])

    dm.prepare_data()
    dm.setup("fit")

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
        gradient_clip_val=0.5,
    )
    trainer.fit(model, datamodule=dm, ckpt_path=checkpoints)

    logger.run.stop()
    return trainer



def test_func(config=None, logger=None, logger_config=None, checkpoints=None):
    if config is None:
        config = default_config

    dm = AudioDataModule(batch_size=config['batch_size'])
    dm.prepare_data()
    dm.setup("test")

    if logger is None and logger_config is not None:
        logger = NeptuneLogger(**logger_config)

    model = HebrewASR(config)

    # log the hyperparameters and not the api key and project name
    logger.run["parameters"] = config

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        logger=logger,
        log_every_n_steps=1,
    )
    trainer.test(model, datamodule=dm, ckpt_path=checkpoints)

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
    