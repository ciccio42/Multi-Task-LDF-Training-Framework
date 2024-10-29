import torch
from functools import partial
from multi_task_il.models.muse.architecture import MUSE

import json

PATH_TO_PT_MODEL = "models/model.pt"
PATH_TO_TF_MODEL = "models/universal-sentence-encoder-multilingual-large-3"

def get_model():
    from multi_task_il.models.muse.tokenizer import get_tokenizer, tokenize

    print(f"get tokenizer...")
    tokenizer = get_tokenizer(PATH_TO_TF_MODEL)
    tokenize = partial(tokenize, tokenizer=tokenizer)

    model_torch = MUSE(
        num_embeddings=128010,
        embedding_dim=512,
        d_model=512,
        num_heads=8,
    )
    model_torch.load_state_dict(
        torch.load(PATH_TO_PT_MODEL)
    )

    return model_torch, tokenize


if __name__ == '__main__':

    model_torch, tokenize = get_model()

    sentence = "Hello, world!"
    res = model_torch(tokenize(sentence))

    print(f"res: {res.shape}")