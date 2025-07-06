import torch
import faiss
from transformers import AutoTokenizer, AutoModel

from utils import show_time


def get_dense_retriever(retriever):
    if retriever == 'contriever':
        query_tokenizer = ctx_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        query_encoder = ctx_encoder = AutoModel.from_pretrained("facebook/contriever")

    #Another Retriever model added here.
    elif retriever == 'minilm':
        query_tokenizer = ctx_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/msmarco-MiniLM-L-6-v3')
        query_encoder = ctx_encoder = AutoModel.from_pretrained('sentence-transformers/msmarco-MiniLM-L-6-v3')
    else:
        raise Exception("Retriever Error")

    return query_tokenizer, ctx_tokenizer, query_encoder, ctx_encoder


def split_batch(instructions, batch_size):
    batch_instructions = []
    sub_batch = []
    for ind, ins in enumerate(instructions):
        if ind != 0 and ind % batch_size == 0:
            batch_instructions.append(sub_batch)
            sub_batch = [ins]
        else:
            sub_batch.append(ins)

    if len(sub_batch) != 0:
        batch_instructions.append(sub_batch)

    return batch_instructions


def get_dense_embedding(instructions, retriever, tokenizer, model, trunc_len=512, batch_size=64):
    emb_list = []
    batch_instructions = split_batch(instructions, batch_size=batch_size)

    for sub_batch in batch_instructions:
        inputs = tokenizer(
            sub_batch,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=trunc_len
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        if retriever == 'contriever':
            # Use mean pooling over last_hidden_state
            def mean_pooling(token_embeddings, mask):
                token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
                return token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]

            embeddings = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
        
        #according to minilm docs, there is no need for mean pooling 
        elif retriever == 'minilm':
            # Use pooler_output
            embeddings = outputs.pooler_output  # shape: [batch_size, hidden_dim]

        else:
            raise ValueError(f"Retriever '{retriever}' not supported.")

        emb_list.append(embeddings.cpu())

    # Stack all batches into final tensor [N, D]
    return torch.cat(emb_list, dim=0)



def dense_neiborhood_search(corpus_data, query_data, metric='ip', num=8):
    if isinstance(query_data, list) :
        xq = torch.vstack(query_data).cpu().numpy()
    else:
        xq = query_data.cpu().numpy()

    if isinstance(corpus_data, list):
        xb = torch.vstack(corpus_data).cpu().numpy()
    else:
        xb = corpus_data.cpu().numpy()
        
    dim = xb.shape[1]
    if metric == 'l2':
        index = faiss.IndexFlatL2(dim)
    elif metric == 'ip':
        index = faiss.IndexFlatIP(dim)
        xq = xq.astype('float32')
        xb = xb.astype('float32')
        faiss.normalize_L2(xq)
        faiss.normalize_L2(xb)
    else:
        raise Exception("Index Metric Not Exist")
    index.add(xb)
    D, I = index.search(xq, num)

    return I[0]


def run_dense_retrieval(query_embedding, ch_text_chunk_embed, ch_text_chunk, chunk_num=4):
    print("{} Dense Retrieval...".format(show_time()))
    neib_ini = dense_neiborhood_search(ch_text_chunk_embed, query_embedding, num=chunk_num)
    neib_ini = list(neib_ini)

    print("{} Retrieved Chunks:".format(show_time()), neib_ini)
    retrieve_text = []
    for ind in neib_ini:
        retrieve_text.append(ch_text_chunk[ind])

    return neib_ini, retrieve_text


if __name__ == '__main__':
    pass
