import os
from transformers import RobertaTokenizerFast


class RobertaTokenizerFastJsonFile:

    @classmethod
    def from_pretrained(self, *args, **kwargs):
        model_path = args[0]
        if model_path[-1] == "/":
            model_path = model_path[:-1]

        if os.path.exists(f"{model_path}/tokenizer.json"):
            return RobertaTokenizerFast(f"{model_path}/vocab.json", f"{model_path}/merges.txt",
                                        tokenizer_file=f"{model_path}/tokenizer.json", **kwargs)
        else:
            return RobertaTokenizerFast(f"{model_path}/vocab.json", f"{model_path}/merges.txt", **kwargs)
