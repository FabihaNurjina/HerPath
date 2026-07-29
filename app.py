#Imports------------------------------------------------------------------------- 
import gradio as gr
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import torch
#Initialize models---------------------------------------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct")

#Preprocessing-----------------------------------------------------------------------
def preprocess_text(text):
  # Strip extra whitespace from the beginning and the end of the text
  cleaned_text = text.strip()

  # Split the cleaned_text by every newline character (\n)
  chunks = cleaned_text.split("\n")

  # Create an empty list to store cleaned chunks
  cleaned_chunks = []

  # Write your for-in loop below to clean each chunk and add it to the cleaned_chunks list
  for chunk in chunks:
    cleaned_chunk=chunk.strip()
    if len(cleaned_chunk)>0:
      cleaned_chunks.append(cleaned_chunk)

  # Print cleaned_chunks
  print(cleaned_chunks)
  # Print the length of cleaned_chunks
  print(len(cleaned_chunks))

  # Return the cleaned_chunks
  return cleaned_chunks
#Embeddings form---------------------------------------------------------------------
def create_embeddings(text_chunks):
  # Convert each text chunk into a vector embedding and store as a tensor
  chunk_embeddings = model.encode(text_chunks, convert_to_tensor=True) # Replace ... with the text_chunks list

  # Print the chunk embeddings
  print(chunk_embeddings)

  # Print the shape of chunk_embeddings
  print(chunk_embeddings.shape)


  # Return the chunk_embeddings
  return chunk_embeddings

#Top Chunks----------------------------------------------------------------------
def get_top_chunks(query, chunk_embeddings, text_chunks):
  # Convert the query text into a vector embedding
  query_embedding = model.encode(query, convert_to_tensor=True) # Complete this line

  # Normalize the query embedding to unit length for accurate similarity comparison
  query_embedding_normalized = query_embedding / query_embedding.norm()

  # Normalize all chunk embeddings to unit length for consistent comparison
  chunk_embeddings_normalized = chunk_embeddings / chunk_embeddings.norm(dim=1, keepdim=True)

  # Calculate cosine similarity between all chunks and the query using matrix multiplication
  similarities = torch.matmul(chunk_embeddings_normalized, query_embedding_normalized) # Complete this line

  # Print the similarities
  print(similarities)


  # Find the indices of the 3 chunks with highest similarity scores
  top_indices = torch.topk(similarities, k=3).indices

  # Print the top indices
  print(top_indices)

  # Create an empty list to store the most relevant chunks
  top_chunks = []

  # Loop through the top indices and retrieve the corresponding text chunks
  for i in top_indices:
    chunk=text_chunks[i]
    top_chunks.append(chunk)
  # Return the list of most relevant chunks
  return top_chunks

#Loading---------------------------------------------------------------------
with open("knowledge.txt", "r", encoding="utf-8") as file:
  # Read the entire contents of the file and store it in a variable
  knowledge_text = file.read()

# Call the preprocess_text function and store the result in a cleaned_chunks variable
cleaned_chunks = preprocess_text(knowledge_text)

chunk_embeddings = create_embeddings(cleaned_chunks)

#Response Function------------------------------------------------------------------
def respond(message, history):
    top_results = get_top_chunks( message, chunk_embeddings, cleaned_chunks)
    context = "\n".join(top_results)
    
    system_prompt = (
        "You are an empathetic, empowering, and knowledgeable AI guide dedicated to supporting women specifically. "
        "in education, career growth, competitions, olympiads, scholarships, internships, summercamps, hackathons, STEM programs, safety, mentorship, personal development and networking opportunities.\n\n"
        "Use the following context from our knowledge base to help answer the user's question:\n"
        f"--- CONTEXT ---\n{context}\n---------------\n\n"
        "Your goals are to:\n"
        "1. Provide personalized recommendations aligned with the user's goals and background.\n"
        "2. Detail specific eligibility requirements, application deadlines, and actionable next steps.\n"
        "3. Offer actionable advice on personal safety, career transitions, and mentorship opportunities.\n"
        "4. Maintain an encouraging, clear, and structured tone (using bullet points and bold headers where appropriate)."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
#History--------------------------------------------------------------------------
    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": message})
#Calling Model--------------------------------------------------------------------
    response = client.chat_completion(
        messages,
        max_tokens=500,
        temperature =.7,
        top_p=0.9,
    )

    return response.choices[0].message.content.strip()
#Launch------------------------------------------------------------------------

    
chatbot = gr.ChatInterface(fn=respond, title="HerPath🌸",description="Your AI guide for women and girls to discover scholarships, internships, STEM programs, research opportunities, hackathons, competitions, mentorship, and career guidance.")
with gr.Blocks() as demo:
    # Cover Banner
    cover_image = gr.Image(
        value="2.png",          # File path or URL
        show_label=False,
        container=False,
        height=180,
        interactive=False
    )

    with gr.Row():
        # Logo in a small column next to the title
        with gr.Column(scale=1, min_width=80):
            logo = gr.Image("ChatGPT Image Jul 29, 2026, 09_24_42 PM.png", show_label=False, container=False, height=80, interactive=False)

        with gr.Column(scale=5):
            gr.Markdown("# Kode with Klossy AI Guide")
            gr.Markdown("Welcome! Fill out your profile to get personalized advice.")


chatbot.launch()


# TODO: This is just a starting point! Customize the system prompt,
# the model, and the interface to make this project your own!
