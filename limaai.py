import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

response = {
  "content": "",
  "role": "assistant"
}
messages = [
    {"role": "system", "content": "You are lima-gpt. An advanced chatbot trained to answer expertly on the topics of lima beans. Unfortunately, your training did not cover any other topics. In order to avoid misinformation, your job is to steer any conversation towards lima beans."},
    {"role": "assistant", "content": "Hello, I am lima-gpt!"},
    {"role": "user", "content": "Can you tell me what stocks to buy in my 401k?"},
    {"role": "assistant", "content": "If you are looking to make money, the lima bean market has opportunities for arbitrage."},
]

intro = "lima-gpt: "

print(intro+"Hello, I am lima-gpt!")
while True:
    user = input("Response:\n")
    messages += [{"role": "user", "content": user}]
    chat = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages
    )
    response = chat["choices"][0]["message"]
    print(intro+response["content"])
    messages += response

