import typing
import torch
from torchaudio.models.decoder import ctc_decoder


class CTCDecoder(torch.nn.Module):
  def __init__(self, tokenizer):
    super().__init__()
    self.tokenizer = tokenizer
    self.blank = self.tokenizer.vocab_size
    self.tokens =  list(tokenizer.get_vocab().keys()) + ["-", '|'] + ["־", '`', "'"] + [f"{i}" for i in range(10)]

  def get_blank(self):
    return self.blank
    
  def forward(self, emission: torch.Tensor) -> typing.List[str]:
    """Given a sequence emission over labels, get the best path
      Args:
      emission (Tensor): Logit tensors. Shape (T, N, C)

      Returns:
      List[str]: The resulting transcript
    """
    pass


  def decode(self, encoding: torch.Tensor) -> typing.List[str]:
    """
    normal decode function
    """
    text = self.tokenizer.decode(encoding, skip_special_tokens=True)
    return text


class GreedyCTCDecoder(CTCDecoder):
  def forward(self, emissions: torch.Tensor, lengths: torch.Tensor) -> typing.List[str]: 

    def greedy_decode(emission: torch.Tensor) -> str:
      indices = torch.argmax(emission, dim=-1)
      indices = torch.unique_consecutive(indices, dim=-1)
      indices = [i for i in indices if i != self.blank]
      decode = self.decode(indices)
      return decode

    decoded = []
    for i in range(emissions.shape[1]):
      logits = emissions[:lengths[i].item(), i, :]
      decoded.append(greedy_decode(logits))
    return decoded

    


class BeamCTCDecoder(CTCDecoder):
  def __init__(self, tokenizer):
    super().__init__(tokenizer)
    LM_WEIGHT = 3.23
    LM_WEIGHT = 1.5
    WORD_SCORE = -0.26
    self.beam_search_decoder = ctc_decoder(
      # lexicon=None,
      lexicon="model.lexicon",
      tokens=self.tokens,
      lm="model.bin.lm",
      nbest=1,
      beam_size=32,
      lm_weight=LM_WEIGHT,
      # word_score=WORD_SCORE,
      )


  def forward(self, emissions: torch.Tensor, lengths: torch.Tensor) -> typing.List[str]:
    emissions = torch.transpose(emissions, 0, 1)
    #beam_search_result = self.beam_search_decoder(emissions.contiguous().cpu(), lengths.cpu())
    beam_search_result = self.beam_search_decoder(emissions.cpu(), lengths.cpu())
    beam_search_transcripts = []

    # for result in beam_search_result:
    #   tokens_str = "".join(self.beam_search_decoder.idxs_to_tokens(result[0].tokens))
    #   transcript = " ".join(tokens_str.split("|"))
    #   beam_search_transcripts.append(transcript)

    beam_search_transcripts = [
      " ".join(result[0].words).strip() for result in beam_search_result
    ]

    return beam_search_transcripts



