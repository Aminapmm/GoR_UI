import argparse

import dgl
from langchain_text_splitters import TokenTextSplitter

from retrieval import *
from utils import *
from prompt_pool import *
from data_process import get_processed_data, split_corpus_by_doc


def rag_retrieval(chunk_list, rag_query, chunk_embedding=None):
    if len(chunk_list) <= RECALL_CHUNK_NUM:
        return chunk_list
    if chunk_embedding is None:
        chunk_embedding = get_dense_embedding(chunk_list, retriever=RETRIEVER, tokenizer=CTX_TOKENIZER,
                                              model=CTX_ENCODER)
    rag_query_embedding = get_dense_embedding([rag_query], retriever=RETRIEVER, tokenizer=QUERY_TOKENIZER,
                                              model=QUERY_ENCODER)
    assert len(rag_query_embedding) == 1
    _, retrieved_text_list = run_dense_retrieval(rag_query_embedding, chunk_embedding, chunk_list,
                                                 chunk_num=RECALL_CHUNK_NUM)

    return retrieved_text_list


def mem_retrieval(mem_chunk_embedding, all_doc_chunk_list, all_doc_chunk_list_embedding, rag_query, graph, retriever,
                  query_tokenizer, query_encoder, recall_chunk_num):
    mem_chunk_list = []
    for node, attrs in graph.nodes(data=True):
        mem_chunk_list.append(node)
    assert len(mem_chunk_embedding) == len(mem_chunk_list), "{}!={}".format(len(mem_chunk_embedding),
                                                                            len(mem_chunk_list))
    mem_chunk_embedding_copy = [i for i in mem_chunk_embedding]
    for chunk, chunk_embedding in zip(all_doc_chunk_list, all_doc_chunk_list_embedding):
        if chunk not in mem_chunk_list:
            mem_chunk_list.append(chunk)
            mem_chunk_embedding_copy.append(chunk_embedding)
    rag_query_embedding = get_dense_embedding([rag_query], retriever=retriever, tokenizer=query_tokenizer,
                                              model=query_encoder)
    mem_chunk_embedding_copy = [i.to(rag_query_embedding[0].device) for i in mem_chunk_embedding_copy]
    assert len(rag_query_embedding) == 1
    assert len(mem_chunk_embedding_copy) == len(mem_chunk_list)
    retrieved_index, retrieved_text_list = run_dense_retrieval(rag_query_embedding, mem_chunk_embedding_copy,
                                                               mem_chunk_list, chunk_num=recall_chunk_num)

    return retrieved_text_list, retrieved_index


def get_node_embedding_list(dgl_graph):
    mem_chunk_embedding = dgl_graph.ndata['feat']
    mem_chunk_embedding = [i for i in mem_chunk_embedding]

    return mem_chunk_embedding


def record_graph_construction(query, support_materials, response, graph, dgl_graph, training_data, answer=None):
    sub_training_data = dict()
    sub_training_data["query"] = query
    if answer:
        sub_training_data["answer"] = answer
    existing_chunks = []
    for node, attrs in graph.nodes(data=True):
        existing_chunks.append(node)
    non_dup_chunks = []
    if response not in existing_chunks:
        non_dup_chunks.append(response)
        graph.add_node(
            response,
        )
        existing_chunks.append(response)
    for chunk in support_materials:
        if chunk not in existing_chunks:
            non_dup_chunks.append(chunk)
            graph.add_node(
                chunk,
            )
            existing_chunks.append(chunk)
    chunk_id_map = dict()
    for chunk_id, chunk in enumerate(existing_chunks):
        chunk_id_map[chunk] = chunk_id
    if len(non_dup_chunks) != 0:
        new_node_embedding = get_dense_embedding(non_dup_chunks, retriever=RETRIEVER, tokenizer=CTX_TOKENIZER,
                                                 model=CTX_ENCODER)
        dgl_graph.add_nodes(num=len(non_dup_chunks), data={'feat': torch.vstack(new_node_embedding).cpu()})
    sub_training_data["response"] = [chunk_id_map[response]]
    sub_training_data["raw"] = []
    for chunk in support_materials:
        sub_training_data["raw"].append(chunk_id_map[chunk])
        if not graph.has_edge(chunk, response):
            graph.add_edge(
                chunk,
                response,
                weight=1
            )
        if not dgl_graph.has_edges_between(chunk_id_map[chunk], chunk_id_map[response]):
            dgl_graph.add_edges(chunk_id_map[chunk],
                                chunk_id_map[response],
                                data={'w': torch.ones(1, 1)})

    training_data.append(sub_training_data)

    return graph, dgl_graph, training_data


def llm2query(prompt, tau=0.5):
    content = get_llm_response_via_ollama(prompt=prompt,
                                       LLM_MODEL=LLM_MODEL,
                                       TAU=tau,
                                       SEED=SEED)
    content = content.split("\n")
    for ind, c in enumerate(content):
        start_ind=0
        for start_ind in range(len(c)):
            if str(c[start_ind]).isalpha():
                start_ind=ind
                break
        content[ind] = c[start_ind:]

    return [i for i in content if len(i.strip()) != 0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument('--train', action='store_true')
    parser.add_argument("--llm_model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0)
    parser.add_argument("--query_tau", type=float, default=0.5)
    parser.add_argument("--retriever", type=str, default="contriever")
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--chunk_overlap", type=int, default=32)
    parser.add_argument("--recall_chunk_num", type=int, default=6)
    parser.add_argument("--query_num", type=int, default=30)
    opt = parser.parse_args()
    DATASET = opt.dataset
    TRAIN = opt.train
    LLM_MODEL = opt.llm_model
    SEED = opt.seed
    TAU = opt.tau
    QUERY_TAU = opt.query_tau
    RETRIEVER = opt.retriever
    CHUNK_SIZE = opt.chunk_size
    CHUNK_OVERLAP = opt.chunk_overlap
    RECALL_CHUNK_NUM = opt.recall_chunk_num
    QUERY_NUM = opt.query_num

    set_seed(int(SEED))
    DEVICE = get_device(int(opt.cuda))

    QUERY_TOKENIZER, CTX_TOKENIZER, QUERY_ENCODER, CTX_ENCODER = get_dense_retriever(retriever=RETRIEVER)
    QUERY_ENCODER = QUERY_ENCODER.to(DEVICE)
    CTX_ENCODER = CTX_ENCODER.to(DEVICE)

    TEXT_SPLITTER = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    data = get_processed_data(dataset=DATASET, train=TRAIN)
    print("{} #Data: {}".format(show_time(), len(data)))
    MAX_NUM = 200 if TRAIN else 30
    data = data[:MAX_NUM]
    check_path("./graph")
    for ind, sample in enumerate(data):

        if TRAIN:
            graph_path = f"./graph/{DATASET}_graph_{ind}.graphml"
            dgl_path = f"./graph/{DATASET}_graph_{ind}.dgl"
            training_data_path = f"./graph/{DATASET}_training_data_{ind}.pkl"
    
            # Check if all output files exist (adjust as needed)
            if os.path.exists(graph_path) and os.path.exists(dgl_path) and os.path.exists(training_data_path):
                print(f"Sample {ind} already processed. Skipping.")
                continue

        else:
            graph_path = f"./graph/{DATASET}_test_graph_{ind}.graphml"
            dgl_path = f"./graph/{DATASET}_test_graph_{ind}.dgl"
    
            # Check if all output files exist (adjust as needed)
            if os.path.exists(graph_path) and os.path.exists(dgl_path):
                print(f"Sample {ind} already processed. Skipping.")
                continue
        
        # Due to budget constraints, we randomly select at most 400 samples for training and 30 samples for evaluation.
        # You can optionally create a dev set for hyper-parameter tuning
        all_doc_chunk_list = split_corpus_by_doc(dataset=DATASET, sample=sample, text_splitter=TEXT_SPLITTER)
        all_doc_chunk_list_embedding = get_dense_embedding(all_doc_chunk_list, retriever=RETRIEVER,
                                                           tokenizer=CTX_TOKENIZER,
                                                           model=CTX_ENCODER)
        graph = nx.Graph()
        dgl_graph = dgl.graph(([], []), num_nodes=0)
        training_data = []
        # Query Simulation
        user_question = []
        user_answer = []
        while len(user_question) < QUERY_NUM:
            unsup_answer = np.random.choice(all_doc_chunk_list, size=1, replace=False)[0].split()
            unsup_answer = " ".join(unsup_answer)
            gen_q = llm2query(prompt=QUERY_GENERATE.format_map({"document": unsup_answer}), tau=QUERY_TAU)[0]
            if gen_q not in user_question:
                user_question.append(gen_q)
                user_answer.append(unsup_answer)
                print("{} Generate Query {}/{}:\n{}".format(show_time(), len(user_question), QUERY_NUM, gen_q))
        # Graph Construction
        for uid, user_query in enumerate(user_question):
            if graph.number_of_nodes() == 0:
                retrieved_chunks = rag_retrieval(chunk_list=all_doc_chunk_list, rag_query=user_query,
                                                 chunk_embedding=all_doc_chunk_list_embedding)
            else:
                mem_chunk_embedding = get_node_embedding_list(dgl_graph=dgl_graph)
                retrieved_chunks, _ = mem_retrieval(mem_chunk_embedding=mem_chunk_embedding, rag_query=user_query,
                                                    graph=graph, all_doc_chunk_list=all_doc_chunk_list,
                                                    all_doc_chunk_list_embedding=all_doc_chunk_list_embedding,
                                                    retriever=RETRIEVER, query_tokenizer=QUERY_TOKENIZER,
                                                    query_encoder=QUERY_ENCODER, recall_chunk_num=RECALL_CHUNK_NUM)
            response = get_llm_response_via_ollama(prompt=QUERY_PROMPT[DATASET].format_map({"question": user_query,
                                                                                         "materials": "\n\n".join(
                                                                                             retrieved_chunks)}),
                                                LLM_MODEL=LLM_MODEL,
                                                TAU=TAU,
                                                SEED=SEED)
            graph, dgl_graph, training_data = record_graph_construction(query=user_query,
                                                                        support_materials=retrieved_chunks,
                                                                        response=response, graph=graph,
                                                                        dgl_graph=dgl_graph,
                                                                        training_data=training_data,
                                                                        answer=user_answer[uid])
            print("{} Graph Construction: {}/{}".format(show_time(), uid + 1, len(user_question)))
            print(dgl_graph)
        # Save
        if TRAIN:
            store_nx(nx_obj=graph, path="./graph/{}_graph_{}.graphml".format(DATASET, ind))
            dgl.save_graphs(filename="./graph/{}_graph_{}.dgl".format(DATASET, ind), g_list=[dgl_graph])
            write_to_pkl(data=training_data, output_file="./graph/{}_training_data_{}.pkl".format(DATASET, ind))
        else:
            store_nx(nx_obj=graph, path="./graph/{}_test_graph_{}.graphml".format(DATASET, ind))
            dgl.save_graphs(filename="./graph/{}_test_graph_{}.dgl".format(DATASET, ind), g_list=[dgl_graph])
