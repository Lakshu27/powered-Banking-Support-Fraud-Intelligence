import groq
client = groq.Groq()
models = client.models.list()
for m in models.data:
    print(m.id)
