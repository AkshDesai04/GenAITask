# Preprocess PDF
import ollama
import preprocessing
import embedding

# Basic initialisations
pdf_file = "file0.pdf"
file_data = preprocessing.extract_text_without_header_footer(pdf_file)
index, chunks = embedding.text_to_vdb(file_data)

# infransing
print("Type your question. Type 'exit' to quit.")
while True:
    user_query = input("User: ").strip()
    if user_query.lower() == "exit":
        break
    context_chunks = embedding.retrieve_relevant_context(user_query, index, chunks)
    context = "\n".join(context_chunks)
    messages = [
        {"role": "system", "content": "You are an AI assistant. Use the following document context to answer the user's question. Context: " + context},
        {"role": "user", "content": user_query}
    ]
    response_stream = ollama.chat(
        model="llama3.2:3b",
        messages=messages,
        stream=True,
    )
    print("model: ", end=" ")
    answer = ""
    for chunk in response_stream:
        content = chunk["message"]["content"]
        print(content, end="", flush=True)
        answer += content
    print()