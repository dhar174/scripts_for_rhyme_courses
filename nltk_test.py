import nltk
from nltk.chat.util import Chat, reflections

from io import StringIO
import sys

set_pairs = [
    [
        r"my name is (.*)",
        ["Hello %1, How are you doing today ?",]
    ],
    [
        r"hi|hey|hello",
        ["Hello", "Hey there",]
    ], 
    [
        r"what is your name?",
        ["You can call me a chatbot ?",]
    ],
    [
        r"how are you ?",
        ["I am fine, thank you! How can i help you?",]
    ],
    [
        r"I am fine, thank you",
        ["great to hear that, how can i help you?",]
    ],
    [
        r"how can i help you? ",
        ["It is my function to help you.", "I just want to fulfill my purpose",]
    ],
    [
        r"i'm doing good",
        ["That's great to hear","How can i help you?:)",]
    ],
    [
        r"(.*) thank you so much, that was helpful",
        ["I am happy to help", "No problem, you're welcome",]
    ],
    [
        r"quit",
    ["Bye, take care. See you soon :) ","It was nice talking to you. See you soon :)"]
],
]




def chatbot():
        print("Hi, I'm your automated assistant") 
        chat = Chat(set_pairs, reflections)
        chat.converse()
       
if __name__ == "__main__":
    chatbot()
