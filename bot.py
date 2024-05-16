import os
from dotenv import load_dotenv
import telebot
import sys
from fashion_clip.fashion_clip import FashionCLIP
import pandas as pd
import numpy as np
from PIL import Image
import pickle


print("Starting...")
load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')
bot = telebot.TeleBot(BOT_TOKEN)


print("Loading the model...")
subset = pd.read_csv('model/subset_data.csv')
text_embeddings = np.load('model/text_embeddings.npy')


# load fclip
fclip = FashionCLIP('fashion-clip')

# with open('model/fclip.pkl', 'rb') as inp:
#     fclip = pickle.load(inp)

# end load fclip


images_dir = 'images'
if not os.path.exists(images_dir):
    os.makedirs(images_dir)


@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    bot.send_message(message.chat.id, "Upload an image of an item of clothes to classify it")


@bot.message_handler(content_types='photo')
def get_image(message):
    try:
        file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        src = os.path.join(images_dir, file_info.file_path.replace('photos/', ''))
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
            new_file.close()

        raw_description = describe_image(src)
        response = format_response(raw_description)
        bot.reply_to(message, response)

    except Exception as e:
        bot.reply_to(message, e)


@bot.message_handler(func=lambda message: message.document.mime_type == 'image/png', content_types=['document'])
def get_document(message):
    try:
        file_info = bot.get_file(message.document.file_id)
        print(file_info)
        downloaded_file = bot.download_file(file_info.file_path)
        src = os.path.join(images_dir, file_info.file_path.replace('documents/', ''))
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
            new_file.close()

        raw_description = describe_image(src)
        response = format_response(raw_description)
        bot.reply_to(message, response)

    except Exception as e:
        bot.reply_to(message, e)


def format_response(tuple):
    return f"This item's group is {tuple[0]}, type is {tuple[2]}, color is {tuple[3]}\nThe description of the closest matching product is \"{tuple[1]}\""


def describe_image(image_path):
    image = Image.open(image_path)

    image_embedding = fclip.encode_images([image], 1)
    image_embedding = image_embedding/np.linalg.norm(image_embedding, ord=2, axis=-1, keepdims=True)
    id_of_matched_object = np.argmax(image_embedding.dot(text_embeddings.T))
    matched_object = subset.iloc[id_of_matched_object]

    return(
        matched_object['product_group_name'], 
        matched_object['detail_desc'],
        matched_object['product_type_name'],
        matched_object['colour_group_name'], 
        matched_object['article_id'], 
        )


print("Listening to requests...")
bot.infinity_polling()