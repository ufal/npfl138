#!/usr/bin/env python3
import argparse

import numpy as np
import torch
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/eleczech-lc-small")
model = transformers.AutoModel.from_pretrained("ufal/eleczech-lc-small", output_hidden_states=True)

dataset = [
    "Podmínkou koexistence jedince druhu Homo sapiens a společenství druhu Canis lupus je sjednocení akustické signální soustavy.",
    "U závodů na zpracování obilí, řízených mytologickými bytostmi je poměrně nízká produktivita práce vyvážena naprostou spolehlivostí.",
    "Vodomilní obratlovci nepatrných rozměrů nejsou ničím jiným, než vodomilnými obratlovci.",
]

print("---Textual tokenization---")
print(*[tokenizer.tokenize(sentence) for sentence in dataset], sep="\n")

print("---Char - subword - word mapping---")
encoded = tokenizer(dataset[0])
print("Token IDs:", encoded.input_ids)
print("Token 2 to chars: {}".format(encoded.token_to_chars(2)))
print("Word 1 to chars: {}".format(encoded.word_to_chars(1)))
print("Word 1 to tokens: {}".format(encoded.word_to_tokens(1)))
print("Char 12 to token: {}".format(encoded.char_to_token(12)))
print("Decoded text: {}".format(tokenizer.decode(encoded.input_ids)))

print("---Running the model---")
batch = tokenizer(dataset, padding="longest")
result = model(torch.as_tensor(batch.input_ids), attention_mask=torch.as_tensor(batch.attention_mask))
print("last_hidden_state: shape {}".format(result.last_hidden_state.shape))
print("hidden_state: shapes", *("{}".format(hidden_state.shape) for hidden_state in result.hidden_states))
