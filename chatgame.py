import openai
import os
import random

openai.api_key = os.getenv("OPENAI_API_KEY")

with open("wordlist.txt", "r") as file:
    wordlist = file.readlines()


def get_word():
    return random.choice(wordlist).strip()

word = get_word()

response = {
  "content": "",
  "role": "assistant"
}
messages = [
    {"role": "system", "content": "We are going to play a game where I will try and get you to say a specific word, without including it in my input. I'll send you a sentence or part of one, and you'll respond with three possibilities for the word, separated by commas. If you can't think of the word, instead of asking for more information or clarification, I want you to give me your best guess or a logical response. Are you ready to play?"},
    {"role": "assistant", "content": "Yes, I am ready to play!"},
]
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
# ]


banned_words = set()

def validate(userstr, word):
    split = userstr.lower().split()
    if word in split:
        print(f"You tried to include the secret word {word} in your input. Try again.")
        return False
    for w in split:
        if w in banned_words:
            print(f"You tried to use \"{w}\", which you've already used before. Try again.")
            return False
    return True

def hit_it(responsestr,word):
    responsestr = responsestr.lower()
    responsestr = ''.join(x for x in responsestr if x.isalpha() or x==" ")
    print(responsestr)
    return word in responsestr.split()

def instructions():
    return f"Secret word is \"{word}\". Banned word list is {banned_words}. Enter your response:"

while True:
    user = input(instructions())
    while not validate(user,word):
        user = input(instructions())

    chat = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages + [{"role": "user", "content": user}]
    )
    response = chat["choices"][0]["message"]
    print(response["content"])
    if hit_it(response["content"],word):
        word = get_word()
        print(f"You did it. Your new word is \"{word}\".")
        banned_words.update(user.lower().split())
        banned_words.remove("_")

