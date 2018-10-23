'''
Task 2
This is the ChatBot which answers if a prediction is good or not

Author: Claudio
'''
import time
import telepot
from telepot.loop import MessageLoop
import pickle

model = pickle.load(open('model.pkl', "rb" ))

def handle(msg):
    """
    A function that will be invoked when a message is
    recevied by the bot
    """
    content_type, chat_type, chat_id = telepot.glance(msg)

    if content_type == "text":
        content = msg["text"]
        prediction = model.predict([content])
        if prediction == 0:
            reply = "This is a bad review."
        elif prediction == 1:
            reply = "This is a good review."
        else:
            reply = "The prediction did not work!"
        bot.sendMessage(chat_id, reply)

if __name__ == "__main__":
    
    # Provide your bot's token
    bot = telepot.Bot("...")
    MessageLoop(bot, handle).run_as_thread()

    while True:
        time.sleep(10)
