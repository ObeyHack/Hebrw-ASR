import typing
import torch
from torchaudio.models.decoder import ctc_decoder


class CTCDecoder(torch.nn.Module):
  def __init__(self, tokenizer):
    super().__init__()
    self.tokenizer = tokenizer
    self.blank = self.tokenizer.vocab_size # 0
    self.tokens =  list(tokenizer.get_vocab().keys()) + ["-", '|'] + ["־", '`']

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
  def forward(self, emission: torch.Tensor) -> typing.List[str]: 
    def greedy_decode(emission: torch.Tensor) -> torch.Tensor:
      indices = torch.argmax(emission, dim=-1)
      indices = torch.unique_consecutive(indices, dim=-1)
      indices = [i for i in indices if i != self.blank]
      decode = self.decode(indices)
      return decode

    decoded = []
    for i in range(emission.shape[1]):
      logits = emission[:, i, :]
      decoded.append(greedy_decode(logits))
    return decoded

    


class BeamCTCDecoder(CTCDecoder):
  def __init__(self, tokenizer):
    super().__init__(tokenizer)
    LM_WEIGHT = 3.23
    WORD_SCORE = -0.26
    self.beam_search_decoder = ctc_decoder(
      lexicon=None,
      tokens=self.tokens,
      lm="model.bin.lm",
      nbest=1,
      beam_size=5,
      lm_weight=LM_WEIGHT,
      word_score=WORD_SCORE,)


  def forward(self, emission: torch.Tensor) -> typing.List[str]:
    emission = torch.transpose(emission, 0, 1)
    lengths = self.input_lengths(emission)
    beam_search_result = self.beam_search_decoder(emission.contiguous().cpu(), lengths.cpu())

    beam_search_transcripts = []

    for result in beam_search_result:
      tokens_str = "".join(self.beam_search_decoder.idxs_to_tokens(result[0].tokens))
      transcript = " ".join(tokens_str.split("|"))
      beam_search_transcripts.append(transcript)

    # beam_search_transcripts = [
    #   " ".join(result[0].words).strip() for result in beam_search_result
    # ]

    return beam_search_transcripts



