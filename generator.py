from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("Loading FLAN-T5 model...")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


def generate_answer(question, context):

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question ONLY using the provided context.

If the answer is not contained in the context, say
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Give a detailed answer.

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.3,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)