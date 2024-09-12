import ExperimentalSemanticChunker as esc

chunker = esc.RecursiveSemanticChunker()
with open("ExampleData/page7.txt", "r", encoding="utf-8") as text_file:
    file_content = text_file.read()
result = chunker.chunk_text(content=file_content, threshold=0.85)
print(result)