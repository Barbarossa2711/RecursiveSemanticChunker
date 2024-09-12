from sentence_transformers import SentenceTransformer
import re
from scipy import spatial
from transformers import AutoTokenizer
import pandas as pd


class SentenceLengthError(Exception):
    def __init__(self, msg):
        super().__init__(msg)
        self.nachricht = msg


class RecursiveSemanticChunker:

    def __df_distance_splitter(self, chunk: pd.DataFrame, max_size=int):
        """
        Recursively splits a DataFrame int Sub-Chunks, which are determined by the biggest
        cosine-distance to the following sentence.
        :param chunk: DataFrame, containing the information about the chunk: 'sentences', 'distance', 'token_count', 'chunk_id'.
        :param max_size: Maximum size of one chunk.
        :return: List of chunks splittet by the biggest cosine-distance until the max size per chunk is reached.
        """
        result = []
        size = []
        if chunk['token_count'].sum() > max_size and chunk.shape[0] != 1:
            chunk.at[chunk.index[-1], 'distance'] = -42
            split_distance = chunk['distance'].max()
            pos_split_distance = chunk.index[chunk['distance'] == split_distance].tolist()[0]
            # prevent infinite loop
            if pos_split_distance != chunk.index[-1]:
                df_top = chunk.iloc[:pos_split_distance + 1].reset_index(drop=True)
                df_bottom = chunk.iloc[pos_split_distance + 1:].reset_index(drop=True)

                if df_top['token_count'].sum() > max_size and not df_top.empty:
                    df_top_result, df_top_size = self.__df_distance_splitter(df_top, max_size)
                    result += df_top_result
                    size += df_top_size
                else:
                    result.append(df_top)
                    size.append(df_top['token_count'].sum())

                if df_bottom['token_count'].sum() > max_size and not df_bottom.empty:
                    df_bottom_result, df_bottom_size = self.__df_distance_splitter(df_bottom, max_size)
                    result += df_bottom_result
                    size += df_bottom_size
                else:
                    result.append(df_bottom)
                    size.append(df_bottom['token_count'].sum())

            else:
                raise SentenceLengthError(f"Sentence is too long (> {max_size} Tokens) and cannot be splittet any"
                                          f"further!")

        elif chunk.shape[0] == 1:
            result.append(chunk)
            size.append(chunk['token_count'].sum())

        else:
            result.append(chunk)
            size.append(chunk['token_count'].sum())

        return result, size

    def __get_max_chunk_length(self, st_model_name: str) -> int:
        """
        Determines the maximum token length of a chunk based on the provided sentence-transformers model name.
        :param st_model_name: Name of a sentence-transformers model.
        :return: Maximum token length of a chunk.
        """
        model = SentenceTransformer(st_model_name)
        tokenizer = model.tokenizer
        max_length = tokenizer.model_max_length

        return max_length

    def chunk_text(self,
                   content: str,
                   threshold: float,
                   regexp_sep=r'[.!?]|\n\n|•',
                   st_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                   max_chunk_size: int = None,
                   ) -> pd.DataFrame:
        """
        Splits the given string into chunks by a threshold until the max chunk size, defined by a sentence-transformers
        model, is reached.
        :param content: String of the text to split.
        :param threshold: Cosine distance threshold for splitting sentences.
        :param regexp_sep: Regular expression to split the text. The default is r'[.!?]|\n\n|•'.
        :param st_model_name: A sentence transformers embedding model name. The default is the all-MiniLM-L6-v2 model.
        :param max_chunk_size: The maximum chunk size wanted. If not specified, it will be determined automatically.
        :return:
        """
        # Prepare input
        sentences = [s for s in re.split(regexp_sep, content) if s]
        sentences_df = pd.DataFrame(sentences, columns=['sentences'])
        if max_chunk_size is None:
            max_chunk_size = self.__get_max_chunk_length(st_model_name=st_model_name)

        # Vectorize sentences
        model = SentenceTransformer(st_model_name)
        sentences_vectors = model.encode(sentences)

        # Calculate cosine distance of one sentences to his successor
        sentences_distance = []
        for i, sentence in enumerate(sentences_vectors):
            if i == len(sentences_vectors) - 1:
                sentences_distance.append(-42)  # dummy distance for the last sentence, because he has no successor
            else:
                distance = spatial.distance.cosine(sentence, sentences_vectors[i + 1])
                sentences_distance.append(distance)
        sentences_df['distance'] = sentences_distance

        # Chunk content by threshold
        chunks = []
        chunks_vectorized = []

        chunk_sentences = []
        chunk_sentences_vectorized = []
        for i, distance in enumerate(sentences_distance):
            # if sentence is similar to his predecessor
            if distance < threshold:
                chunk_sentences.append(sentences[i])
                chunk_sentences_vectorized.append(sentences_vectors[i])
            # if sentence is not similar to his predecessor
            else:
                # if current chunk has no sentences in it
                if not chunk_sentences:
                    # Add current sentence to the empty chunk
                    chunk_sentences.append(sentences[i])
                    chunk_sentences_vectorized.append(sentences_vectors[i])
                    # Add current chunk to the chunk-list
                    chunks.append(chunk_sentences.copy())
                    chunks_vectorized.append(chunk_sentences_vectorized.copy())
                    # clear current chunk lists
                    chunk_sentences = []
                    chunk_sentences_vectorized = []
                else:
                    # Add current chunk to the chunk lists
                    chunks.append(chunk_sentences.copy())
                    chunks_vectorized.append(chunk_sentences_vectorized.copy())
                    # Put the sentence that's too different for the chunk into a new chunk
                    chunk_sentences = [sentences[i]]
                    chunk_sentences_vectorized = [sentences_vectors[i]]
        # Add the last chunk to the chunk list
        chunks.append(chunk_sentences)
        chunks_vectorized.append(chunk_sentences_vectorized)

        # Calculate the token count of the sentences and give the sentences a chunk-id
        tokenizer = AutoTokenizer.from_pretrained(st_model_name)
        chunk_sizes = []  # List for size of chunks with the corresponding id
        for i, chunk in enumerate(chunks):
            chunk_token_size = 0
            for sentence in chunk:
                tokens = tokenizer.tokenize(sentence)
                chunk_token_size += len(tokens)
                sentences_df.loc[sentences_df['sentences'] == sentence, 'token_count'] = len(tokens)
                sentences_df.loc[sentences_df['sentences'] == sentence, 'chunk_id'] = i
            chunk_sizes.append(chunk_token_size)

        # Identify the id the chunks that are too big for the chosen model
        problematic_chunk_ids = [index for index, value in enumerate(chunk_sizes) if value > max_chunk_size]

        # Split problematic Chunks recursively
        # Get the valid chunks before the first problematic chunk
        valid_chunk_before = []
        for problematic_id in range(min(problematic_chunk_ids)):
            problematic_id_df = sentences_df.loc[sentences_df["chunk_id"] == problematic_id].reset_index(drop=True)
            valid_chunk_before.append(problematic_id_df)

        resulting_chunks = valid_chunk_before.copy()
        index_valid_chunk_after = -1
        # recursively split the problematic chunk and create a new index
        for current_problematic_chunk_id in problematic_chunk_ids:
            last_index = -1
            df_current_chunk = sentences_df.loc[sentences_df["chunk_id"] == current_problematic_chunk_id].reset_index(
                drop=True)
            results, _ = self.__df_distance_splitter(chunk=df_current_chunk, max_size=max_chunk_size)
            for j, i in enumerate(range(current_problematic_chunk_id, current_problematic_chunk_id + len(results))):
                results[j]['chunk_id'] = i
                last_index = i
            resulting_chunks += results
            index_valid_chunk_after = last_index + 1

        valid_chunk_after = []
        for index, id in enumerate(range(max(problematic_chunk_ids) + 1, len(chunk_sizes))):
            id_df = sentences_df.loc[sentences_df['chunk_id'] == id].reset_index(drop=True)
            id_df['chunk_id'] = index_valid_chunk_after + index
            valid_chunk_after.append(id_df)
        # combine the new chunks with the valid chunks
        resulting_chunks += valid_chunk_after
        correct_chunks = pd.concat(resulting_chunks, ignore_index=True)

        return correct_chunks
