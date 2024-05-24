import os
# from dotenv import load_dotenv
import telebot
import numpy as np
from keras.models import load_model
from keras.preprocessing import image


print("Starting...")
# load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')
bot = telebot.TeleBot(BOT_TOKEN)


print("Loading the model...")
model = load_model('model/vgg16_model.h5')
classes = ['dress',
            'hat',
            'longsleeve',
            'outwear',
            'pants',
            'shirt',
            'shoes',
            'shorts',
            'skirt',
            't-shirt']
print("Model loaded!")


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

        raw_description = classify(src)
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

        raw_description = classify(src)
        response = format_response(raw_description)
        bot.reply_to(message, response)

    except Exception as e:
        bot.reply_to(message, e)


def format_response(_class):
    return f"The detected clothes class is {_class}"


def classify(image_path):
    print(f"Processing the image {image_path}")
    image_data = image.load_img(image_path, color_mode ='rgb', target_size = (224, 224))
    image_data = image.img_to_array(image_data)
    image_data = np.expand_dims(image_data, axis = 0)
    result = model.predict(image_data)
    res = np.argmax(result)
    print("Image processed!")
    return classes[res]


print("Listening to requests...")
bot.infinity_polling()