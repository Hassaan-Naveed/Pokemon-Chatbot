#########################################################
# Pokemon Chatbot
# AI Coursework - Hassaan Naveed
#########################################################

# Initialise libraries
import json
import requests
import aiml
from nltk.corpus import stopwords
from nltk.sem import Expression
from nltk.inference import ResolutionProver
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
from tensorflow import keras
from PIL import Image
import numpy as np
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import os

# Initialise aiml agent
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="chatbot-answers.xml")

# Initialise NLTK interface and knowledgebase
read_expr = Expression.fromstring
kb = []
data = pd.read_csv('kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]

# Check knowledgebase integrity
integrity_check = ResolutionProver().prove(None, kb, verbose=False)
if integrity_check:
    print("Error, contradiction found in kb")
    quit()

# Initialise Speech Recogniser
r = sr.Recognizer()

# Initialise API URLs
api_url = r"https://pokeapi.co/api/v2/"
pokemon_ext = r"pokemon/"
types_ext = r"type/"
species_ext = r"pokemon-species/"

# Initialise Azure Vision API
cog_key = '04103935cb154f6d8692b0b2b29cf318'
cog_endpoint = 'https://coursework-task-d.cognitiveservices.azure.com/'

# Initialise similarity dataframe
df = pd.read_csv("QA.csv")

# Labels
all_types = ["normal", "fire", "water", "grass", "electric", "ice", "fighting", "poison", "ground",
             "flying", "psychic", "bug", "rock", "ghost", "dark", "dragon", "steel", "fairy"]

labels = [('Abra', 0), ('Aerodactyl', 1), ('Alakazam', 2), ('Alolan Sandslash', 3), ('Arbok', 4), ('Arcanine', 5),
          ('Articuno', 6), ('Beedrill', 7), ('Bellsprout', 8), ('Blastoise', 9), ('Bulbasaur', 10), ('Butterfree', 11),
          ('Caterpie', 12), ('Chansey', 13), ('Charizard', 14), ('Charmander', 15), ('Charmeleon', 16), ('Clefable', 17),
          ('Clefairy', 18), ('Cloyster', 19), ('Cubone', 20), ('Dewgong', 21), ('Diglett', 22), ('Ditto', 23),
          ('Dodrio', 24), ('Doduo', 25), ('Dragonair', 26), ('Dragonite', 27), ('Dratini', 28), ('Drowzee', 29),
          ('Dugtrio', 30), ('Eevee', 31), ('Ekans', 32), ('Electabuzz', 33), ('Electrode', 34), ('Exeggcute', 35),
          ('Exeggutor', 36), ('Farfetchd', 37), ('Fearow', 38), ('Flareon', 39), ('Gastly', 40), ('Gengar', 41),
          ('Geodude', 42), ('Gloom', 43), ('Golbat', 44), ('Goldeen', 45), ('Golduck', 46), ('Golem', 47),
          ('Graveler', 48), ('Grimer', 49), ('Growlithe', 50), ('Gyarados', 51), ('Haunter', 52), ('Hitmonchan', 53),
          ('Hitmonlee', 54), ('Horsea', 55), ('Hypno', 56), ('Ivysaur', 57), ('Jigglypuff', 58), ('Jolteon', 59),
          ('Jynx', 60), ('Kabuto', 61), ('Kabutops', 62), ('Kadabra', 63), ('Kakuna', 64), ('Kangaskhan', 65),
          ('Kingler', 66), ('Koffing', 67), ('Krabby', 68), ('Lapras', 69), ('Lickitung', 70), ('Machamp', 71),
          ('Machoke', 72), ('Machop', 73), ('Magikarp', 74), ('Magmar', 75), ('Magnemite', 76), ('Magneton', 77),
          ('Mankey', 78), ('Marowak', 79), ('Meowth', 80), ('Metapod', 81), ('Mew', 82), ('Mewtwo', 83),
          ('Moltres', 84), ('MrMime', 85), ('Muk', 86), ('Nidoking', 87), ('Nidoqueen', 88), ('Nidorina', 89),
          ('Nidorino', 90), ('Ninetales', 91), ('Oddish', 92), ('Omanyte', 93), ('Omastar', 94), ('Onix', 95),
          ('Paras', 96), ('Parasect', 97), ('Persian', 98), ('Pidgeot', 99), ('Pidgeotto', 100), ('Pidgey', 101),
          ('Pikachu', 102), ('Pinsir', 103), ('Poliwag', 104), ('Poliwhirl', 105), ('Poliwrath', 106), ('Ponyta', 107),
          ('Porygon', 108), ('Primeape', 109), ('Psyduck', 110), ('Raichu', 111), ('Rapidash', 112), ('Raticate', 113),
          ('Rattata', 114), ('Rhydon', 115), ('Rhyhorn', 116), ('Sandshrew', 117), ('Sandslash', 118), ('Scyther', 119),
          ('Seadra', 120), ('Seaking', 121), ('Seel', 122), ('Shellder', 123), ('Slowbro', 124), ('Slowpoke', 125),
          ('Snorlax', 126), ('Spearow', 127), ('Squirtle', 128), ('Starmie', 129), ('Staryu', 130), ('Tangela', 131),
          ('Tauros', 132), ('Tentacool', 133), ('Tentacruel', 134), ('Vaporeon', 135), ('Venomoth', 136),
          ('Venonat', 137), ('Venusaur', 138), ('Victreebel', 139), ('Vileplume', 140), ('Voltorb', 141),
          ('Vulpix', 142), ('Wartortle', 143), ('Weedle', 144), ('Weepinbell', 145), ('Weezing', 146),
          ('Wigglytuff', 147), ('Zapdos', 148), ('Zubat', 149)]

# Welcome user
print("Hi! Feel free to ask any questions about a Pokemon! Type 'VOICE' to ask a question with your microphone!")


# Image Classifier
def classifier(path):
    img = Image.open(path)
    img = img.resize((128, 128))

    img = np.array(img)
    if img.shape[2] != 3:
        img = img[:, :, :3]

    img = img / 255
    img_array = np.expand_dims(img, axis=0)

    model = keras.models.load_model("pokemon_classifier.h5")

    pred_class = np.argmax(model.predict(img_array), axis=-1)
    poke_class = {value: key for key, value in labels}

    print("That is " + poke_class[pred_class[0]])


# Get Azure Vision Description
def vision(path):
    # Get a client for the computer vision service
    computervision_client = ComputerVisionClient(cog_endpoint, CognitiveServicesCredentials(cog_key))

    # Get a description from the computer vision service
    image_stream = open(path, "rb")
    description = computervision_client.describe_image_in_stream(image_stream)

    caption_text = ""

    if len(description.captions) == 0:
        caption_text = 'No caption detected'
    else:
        for caption in description.captions:
            caption_text = caption_text + " '{}'\n(Confidence: {:.2f}%)".format(caption.text, caption.confidence * 100)

    print(caption_text)


# Calculate TF_DIF score
def tf_idf(query):
    query = [query]
    v = TfidfVectorizer(stop_words=stopwords.words('english'))
    similarity_index_list = cosine_similarity(v.fit_transform(df['Question']), v.transform(query)).flatten()

    if not all(i == 0 for i in similarity_index_list):
        answer = df.loc[similarity_index_list.argmax(), "Answer"]
        print(answer)
    else:
        print("Sorry, I did not understand your question!")


# Get from API
def api_get(url_ext, inp):
    succeeded = False
    response = requests.get(api_url + url_ext + inp)
    if response.status_code == 200:
        response_json = json.loads(response.content)
        if response_json:
            succeeded = True
            return response_json
    if not succeeded:
        print("Sorry, I didn't understand what you asked!")
        return ""


# Get pokemon type
def get_types(response_json):
    types = []
    for i in range(len(response_json["types"])):
        types.append(response_json["types"][i]["type"]["name"])

    return types


# Get pokemon weaknesses
def get_weakness(response_json):
    weakness = []
    for y in range(len(response_json["damage_relations"]["double_damage_from"])):
        weakness.append(response_json["damage_relations"]["double_damage_from"][y]["name"])

    return weakness


# Get pokemon flavour text
def get_flavour_text(response_json):
    return response_json['flavor_text_entries'][0]['flavor_text']


# Get voice input
def voice_input():
    try:
        with sr.Microphone() as source:
            print("Say Something!")
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio = r.listen(source)
            voiceInp = r.recognize_google(audio)
            return voiceInp.lower()

    except sr.RequestError as e:
        print("Sorry, I couldn't process your voice query!")
    except sr.UnknownValueError:
        print("Sorry, I was unable to recognise your speech!")


# Main loop
def main():
    while True:
        # Get user input
        try:
            userInput = input("> ")
        except (KeyboardInterrupt, EOFError) as e:
            print("Bye!")
            break

        if userInput == "VOICE":
            userInput = voice_input()
            if userInput is None:
                break
            else:
                print(">", userInput)

        # Preprocess the input and determine response agent
        responseAgent = "aiml"

        # Activate selected response agent
        if responseAgent == "aiml":
            answer = kern.respond(userInput)

            # Postprocess answer for commands
            if answer[0] == '{':
                params = answer[1:].split('}')
                cmd = int(params[0])

                # Goodbye Message
                if cmd == 0:
                    print(params[1])
                    break

                # Types
                elif cmd == 1:
                    response_json = api_get(pokemon_ext, params[1])
                    if response_json == "":
                        break

                    types = get_types(response_json)
                    print(params[1], "is", ", ".join(str(i) for i in types))

                # Super effective against
                elif cmd == 2:
                    # Types
                    if params[1] in all_types:
                        response_json = api_get(types_ext, params[1])
                        if response_json == "":
                            break

                        weakness = get_weakness(response_json)
                        print(params[1], "types are weak to", ", ".join(str(i) for i in weakness), "type moves")

                    # Pokemon
                    else:
                        response_json = api_get(pokemon_ext, params[1])
                        if response_json == "":
                            break

                        types = get_types(response_json)
                        all_weakness = []

                        for i in range(len(types)):
                            response_json = api_get(types_ext, types[i])
                            if response_json == "":
                                break

                            weakness = get_weakness(response_json)
                            all_weakness.extend(weakness)

                        print(params[1], "is weak to", ", ".join(str(i) for i in all_weakness), "type moves")

                # Pokedex Entry
                elif cmd == 3:
                    response_json = api_get(species_ext, params[1])
                    print(get_flavour_text(response_json))

                # Image Classifier
                elif cmd == 4:
                    if (os.path.exists(params[1] + ".png")):
                        classifier(params[1] + ".png")
                    else:
                        print("Sorry, I could not find that file!")

                # Azure Vision
                elif cmd == 5:
                    if (os.path.exists(params[1] + ".png")):
                        vision(params[1] + ".png")
                    else:
                        print("Sorry, I could not find that file!")

                # Add to kb
                elif cmd == 97:
                    object, subject = params[1].split(' is ')
                    expr = read_expr(subject + '(' + object + ')')
                    kb.append(expr)

                    integrity_check = ResolutionProver().prove(None, kb, verbose=False)

                    if not integrity_check:
                        print("Okay! I'll remember that", object, "is", subject)
                    else:
                        print("Sorry! This contradicts with what i know!")
                        kb.pop()

                # Check kb
                elif cmd == 98:
                    object, subject = params[1].split(' is ')
                    expr = read_expr(subject + '(' + object + ')')

                    answer = ResolutionProver().prove(expr, kb, verbose=False)
                    if answer:
                        print('Correct.')
                    else:
                        kb.append(expr)
                        integrity_check = ResolutionProver().prove(None, kb, verbose=False)

                        if integrity_check:
                            print("Incorrect")
                        else:
                            print("Sorry, I don't know :(")
                        kb.pop()

                # Unresolved Input
                elif cmd == 99:
                    tf_idf(userInput)
            else:
                print(answer)


main()
