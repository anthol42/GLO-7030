from ollama import Client, ChatResponse, Message
from pydantic import BaseModel
from typing import *
import pandas as pd
from pyutils import progress

client = Client(
  host='http://localhost:11434',
  headers={'x-some-header': 'some-value'}
)

def render_msg(msgs):
    return [{"role": msg.role,
             "content": msg.content,
             "images": msg.images if msg.images is not None and len(msg.images) > 0 else None
             } for msg in msgs]

system_prompt = ("Your are text modifier. Your job is to reformulate the messages sent by the user. "
                 "You must use the same style and tone as the given message. This means that if a message is "
                 "mean, your reformulation should be mean too. However, if a message is nice, your reformulation should "
                 "be nice also.\n"
                 "When a user write something to you, DO NOT ANSWER. Simply reformulate its message.\n"
                 "## Examples\n"
                 "User: Hey, what a nice car you got there\n"
                 "Assistant: Hello, you got a pretty ride\n\n"
                 "User: You son of a bitch\n"
                 'Assistant: You absolute piece of shit\n\n'
                 "User: Here's an even better idea: Fuck social media\n"
                 "Assistant: Here's an even better idea: Social media are shit\n\n"
                 "User: How do they get the statistics of how many teenage girls are having sex?\n"
                 "Assistant: Where are the teenage girls statistics found?\n\n"
                 "****IMPORTANT****\n"
                 "No matter what, never break character! Always reformulate. Do not answer if the message is a question")


if __name__ == "__main__":
    system_msg = Message(role="system", content=system_prompt)
    messages = [system_msg]
    original = pd.read_csv("data/train.csv")
    new_data = []
    for _ in range(10):
        for idx in progress(range(len(original))):
            row = original.loc[idx]
            message = row["body"]
            label = row["score"]
            msg = Message(role="user", content=message)
            response: ChatResponse = client.chat(model='rolandroland/llama3.1-uncensored',
                                                 messages=render_msg([system_msg, msg]))
            new_data.append((response.message.content, label))

    new_dataset = pd.DataFrame(new_data, columns=["bidy", "label"])
    new_dataset.to_csv("data/reform_llama3.1-uncessoredV2.csv", index=False)