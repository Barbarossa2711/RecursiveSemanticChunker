# RecursiveSemanticChunker
Experimental approach of a semantic chunker that can limit the maximum chunk size.

## Semantic chunking
Semantic Chunking is a concept introduced by Greg Kamradt in his notebook [5_Levels_Of_Text_Splitting](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb).
The idea of a semantic chunker is to split sentences based on their similarity.
This creates semantically coherent chunks.

But if your resources are limited, you may want to use smaller embedding
models such as sentence-transformers/all-MiniLM-L6-v2, which truncates any content after the 256 token.
Because of this small context length, a semantic chunker like langchain implemented [here](https://python.langchain.com/v0.2/api_reference/experimental/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html) is not very suitable.
This chunker does not care about a maximum chunk size, it just splits the given content based on its contextual
connection.

The RecursiveSemanticChunker presented in the following tries to recursively split the chunk that is too big for a
chosen model into smaller chunks based on the maximum cosine distance within the problematic chunk.

## How does the RecursiveSemanticChunker work?
*(The explanation is based on the example page7.txt, which can be found in ExampleData)*

First, the chunker splits the content into blocks and calculates the cosine distance from one block to its successor.
Using an example threshold of 0.85, this results in the following figure.

![dist sentences](Figures/dist_sentences.pdf)

The red line represents the defined threshold and the blue points are the cosine distances of the sentences.
By crossing the red line, a sentence ends and a new one begins. This is the standard procedure for semantic chunking.

The resulting chunk sizes are shown in the following figure:

![chunk sizes](Figures/chunk_sizes.pdf)

As illustrated in the preceding example, the data chunk with the identifier '4' exceeds the capacity of the
*all-MiniLM-L6-v2* embedding model for complete processing.

To address this issue, the RecursiveSemanticChunker examines the problematic chunks and divides them into smaller units.
The point of division is determined by the maximum cosine distance within the chunk. By doing so, we ensure that the
semantic splitting is further refined. This process will continue until the largest resulting chunk is smaller than
the maximum context length of the model.

The figure below illustrates the cosine distances within the chunk 4:

![dist problematic chunk](Figures/dist_problematic_chunk.pdf)

Upon completion of the process, the chunk with the identifier "4" is divided into five smaller chunks, as illustrated
in the figure below.

![chunk sizes splitted](Figures/chunk_sizes_splitted.pdf)

## [Note]
This is an initial experimental approach to the concept of recursively dividing text into progressively smaller units
until a specified size is reached. The concept was developed as part of a Bachelor's thesis at the Hochschule
Niederrhein, in which the objective was to construct an open-source RAG system and evaluate its performance in
comparison to gpt-4o.\
To illustrate, the code was not optimised for rapid execution and merely serves as a foundation for further work. Due
to time constraints, the concept was not subjected to further elaboration in the Bachelor's thesis.
