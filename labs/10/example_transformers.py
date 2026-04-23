#!/usr/bin/env python3
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Suppress the LOAD REPORT with weight discrepancies.

import torch
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/eleczech-lc-small")
model = transformers.AutoModel.from_pretrained("ufal/eleczech-lc-small", output_hidden_states=True)

dataset = [
    "Podmínkou koexistence jedince druhu Homo sapiens a společenství druhu Canis lupus je sjednocení akustické signální soustavy.",
    "U závodů na zpracování obilí, řízených mytologickými bytostmi je poměrně nízká produktivita práce vyvážena naprostou spolehlivostí.",
    "Vodomilní obratlovci nepatrných rozměrů nejsou ničím jiným, než vodomilnými obratlovci.",
]

print("\n---Textual tokenization---")
print(*[tokenizer.tokenize(sentence) for sentence in dataset], sep="\n")

print("\n---Char - subword - word mapping---")
encoded = tokenizer(dataset[0])
print("Token IDs:", encoded.input_ids)
print(f"Token 2 to chars: {encoded.token_to_chars(2)}")
print(f"Word 1 to chars: {encoded.word_to_chars(1)}")
print(f"Word 1 to tokens: {encoded.word_to_tokens(1)}")
print(f"Char 12 to token: {encoded.char_to_token(12)}")
print(f"Decoded text: {tokenizer.decode(encoded.input_ids)}")

print("\n---Running the model---")
batch = tokenizer(dataset, padding="longest")
result = model(torch.as_tensor(batch.input_ids), attention_mask=torch.as_tensor(batch.attention_mask))
print(f"last_hidden_state: shape {result.last_hidden_state.shape}")
print("hidden_state: shapes", *(f"{hidden_state.shape}" for hidden_state in result.hidden_states))
