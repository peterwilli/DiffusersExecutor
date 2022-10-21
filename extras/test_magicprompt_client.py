from jina import Client, Document

c = Client(host='http://localhost:5000')
print(c.post('/magic_prompt/stable_diffusion', Document(text='Laughing mother in Greece'), parameters = {})[0].text)
