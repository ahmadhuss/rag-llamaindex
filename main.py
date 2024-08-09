import os
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.response.pprint_utils import pprint_response
# To get responses based on our request
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
# Persistent Storage
from llama_index.core import StorageContext, load_index_from_storage

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
PERSIST_DIR = './storage'


def run():
    documents = SimpleDirectoryReader('data').load_data()
    print('Documents MetaData:', documents)
    #  Generate index embeddings inside in memory
    # index = VectorStoreIndex.from_documents(documents, show_progress=True)
    # print('Create Index in memory:', index)

    # ============================ Retriever ===============================
    # Retriever Instantiation => Get 4 responses instead of by default 2
    # retriever = VectorIndexRetriever(index=index, similarity_top_k=4)
    # Similarity postprocessor helps us to get the result based on our similarity cutoff give me above 80
    # postprocessor = SimilarityPostprocessor(similarity_cutoff=0.80)
    # Prepare the Query Engine with Retriever
    # query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[postprocessor])
    # ============================ Retriever ===============================

    # ============================ Storage ===============================
    index = persistent_storage()
    # ============================ Storage ===============================

    # Prepare the Query Engine
    # LlamaIndex uses OpenAI's gpt-3.5-turbo by default.
    # Make sure your API key is available to your code by setting
    # it as an environment variable.
    query_engine = index.as_query_engine()
    # Pass a query to index and get the most suitable response from the LLamaIndexing embeddings.
    # means first response in the list
    # response = query_engine.query('What is News project all about?')
    response = query_engine.query('What is CCSC In Singapore?')
    # print('Normal Print Query Response:', response)
    pprint_response(response, show_source=True)


def persistent_storage():
    if not os.path.exists(PERSIST_DIR):
        # load the documents and create the index
        documents = SimpleDirectoryReader('data').load_data()
        print('Documents MetaData:', documents)
        #  Generate index embeddings inside in memory
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        print('Create Index in memory:', index)
        # Store it for later
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        return index
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        #  Load the storage index
        index = load_index_from_storage(storage_context)
        print('Index has been loaded from Storage directory:', index)
        return index


if __name__ == '__main__':
    run()
