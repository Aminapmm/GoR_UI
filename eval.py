import argparse

import dgl
from langchain_text_splitters import TokenTextSplitter

from retrieval import *
from utils import *
from prompt_pool import *
from data_process import get_processed_data, split_corpus_by_doc, eval_data_generation
from graph_construction import mem_retrieval
from training_preparation import integrate_isolated
from train import GoR


def infer_node_embedding(dgl_graph, model_path):
    model = GoR(in_dim=IN_DIM, num_hidden=HIDDEN_DIM, num_layer=NUM_LAYER, n_head=N_HEAD)
    model.load_state_dict(torch.load(model_path))
    model = model.encoder
    model.eval()
    model.to(DEVICE)
    dgl_graph = dgl_graph.to(DEVICE)
    dgl_graph = dgl.add_self_loop(dgl_graph)
    node_embedding = model(dgl_graph, dgl_graph.ndata['feat']).detach()
    node_embedding = [i for i in node_embedding]

    return node_embedding


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--llm_model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0)
    parser.add_argument("--retriever", type=str, default="contriever")
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--chunk_overlap", type=int, default=32)
    parser.add_argument("--recall_chunk_num", type=int, default=6)
    parser.add_argument("--in_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--eval_num", type=int , default=30)
    opt = parser.parse_args()
    DATASET = opt.dataset
    LLM_MODEL = opt.llm_model
    SEED = opt.seed
    TAU = opt.tau
    RETRIEVER = opt.retriever
    CHUNK_SIZE = opt.chunk_size
    CHUNK_OVERLAP = opt.chunk_overlap
    RECALL_CHUNK_NUM = opt.recall_chunk_num
    IN_DIM = opt.in_dim
    HIDDEN_DIM = opt.hidden_dim
    NUM_LAYER = opt.num_layer
    N_HEAD = opt.n_head

    #Now User Can Specify the number of queries for evaluation
    EVAL_NUM = opt.eval_num

    set_seed(int(SEED))
    DEVICE = get_device(int(opt.cuda))

    QUERY_TOKENIZER, CTX_TOKENIZER, QUERY_ENCODER, CTX_ENCODER = get_dense_retriever(retriever=RETRIEVER)
    QUERY_ENCODER = QUERY_ENCODER.to(DEVICE)
    CTX_ENCODER = CTX_ENCODER.to(DEVICE)

    TEXT_SPLITTER = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    data = get_processed_data(dataset=DATASET, train=False)
    print("{} #Data: {}".format(show_time(), len(data)))
    data = data[:EVAL_NUM]
    check_path("./graph")
    check_path("./result")
    result_recorder = dict()
    for ind, sample in enumerate(data):
        all_doc_chunk_list = split_corpus_by_doc(dataset=DATASET, sample=sample, text_splitter=TEXT_SPLITTER)
        all_doc_chunk_list_embedding = get_dense_embedding(all_doc_chunk_list, retriever=RETRIEVER,
                                                           tokenizer=CTX_TOKENIZER,
                                                           model=CTX_ENCODER)
        graph = load_nx(path="./graph/{}_test_graph_{}.graphml".format(DATASET, ind))
        gs, _ = dgl.load_graphs("./graph/{}_test_graph_{}.dgl".format(DATASET, ind))
        dgl_graph = gs[0]
        graph, dgl_graph, = integrate_isolated(graph=graph, dgl_graph=dgl_graph, all_doc_chunk_list=all_doc_chunk_list,
                                               all_doc_chunk_list_embedding=all_doc_chunk_list_embedding)
        check_path("./weights")
        mem_chunk_embedding = infer_node_embedding(dgl_graph=dgl_graph, model_path="./weights/{}.pth".format(DATASET))
        eval_data = eval_data_generation(dataset=DATASET, sample=sample)
        for test_query in eval_data:
            retrieved_chunks, _ = mem_retrieval(mem_chunk_embedding=mem_chunk_embedding,
                                                rag_query=test_query["rag_query"],
                                                graph=graph,
                                                all_doc_chunk_list=all_doc_chunk_list,
                                                all_doc_chunk_list_embedding=all_doc_chunk_list_embedding,
                                                retriever=RETRIEVER,
                                                query_tokenizer=QUERY_TOKENIZER,
                                                query_encoder=QUERY_ENCODER,
                                                recall_chunk_num=RECALL_CHUNK_NUM)
            response = get_llm_response_via_ollama(
                prompt=QUERY_PROMPT_NORMAL[DATASET].format_map({"question": test_query["query"],
                                                                "materials": "\n\n".join(
                                                                    retrieved_chunks)}),
                LLM_MODEL=LLM_MODEL,
                TAU=TAU,
                SEED=SEED)
            # print(text_wrap("LLM RESPONSE:\n"), response)
            # print(text_wrap("GOLDEN ANSWER: {}".format(test_query["summary"])))
            result_recorder[str(ind) + test_query['query']] = {"response": response, "gt": test_query["summary"]}

    write_to_json(result_recorder, "./result/{}.json".format(DATASET))
