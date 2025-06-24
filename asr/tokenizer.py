from tokenizers import Tokenizer, models

my_tokenizer = Tokenizer(models.BPE())
my_tokenizer.add_special_tokens(["▢"])
my_tokenizer.add_tokens(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ '"))
blank_id = my_tokenizer.token_to_id("▢")
