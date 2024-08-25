import typing
import torch


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, tokenizer, blank=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.skip_ids = 7
        self.blank = tokenizer.vocab_size - self.skip_ids

    def forward(self, emission: torch.Tensor) -> typing.List[str]:
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]

        indices = [i + self.skip_ids for i in indices]   
        decode =  self.decode(indices)
        return decode
      

    def decode(self, encoding: torch.Tensor) -> typing.List[str]:
      """
      normal decode function
      """
      text = self.tokenizer.decode(encoding, skip_special_tokens=True)
      return text



