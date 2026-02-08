import os
from dotenv import load_dotenv
from google import genai

import numpy as np
from gpt_sovits.infer import GPTSoVITSInference
from scipy.io import wavfile

import sounddevice as sd

load_dotenv()

inference = GPTSoVITSInference(
    bert_path="pretrained_models/chinese-roberta-wwm-ext-large",
    cnhubert_base_path="pretrained_models/chinese-hubert-base",
    is_half=False,
    device="cpu"
)

inference.load_sovits(os.path.join("pretrained_models/anon1_e8_s2184.pth"))
inference.load_gpt(
    os.path.join("pretrained_models/anon1-e15.ckpt")
)

inference.set_prompt_audio(
    prompt_audio_path=os.path.join("pretrained_models/んー、指痛い練習つらい.wav"),
    prompt_text="んー、指痛い練習つらい"
)

import sounddevice as sd
import numpy as np
import queue

rms_queue = queue.Queue()

text_queue = queue.Queue()

def play_streaming_tts(tts, text_input):
    stream = None
    try:
        for sample_rate, audio_chunk, text in tts.get_tts_wav_stream(text_input.lower()):
            if stream is None:
                stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype='float32')
                stream.start()

            audio_chunk = np.asarray(audio_chunk, dtype=np.float32) / 32768.0

            rms = np.sqrt(np.mean(audio_chunk ** 2))

            try:
                rms_queue.put_nowait(rms)
            except queue.Full:
                pass

            try:
                text_queue.put_nowait(text)
            except queue.Full:
                pass

            stream.write(audio_chunk)

    finally:
        if stream is not None:
            stream.stop()
            stream.close()

import queue
import json
import time
from vosk import Model, KaldiRecognizer

SAMPLE_RATE = 16000
NO_WORDS_DURATION = 1.5

# print("Loading Vosk model...")
# rec_model = Model(os.path.join("vosk-model-en-us-0.22"))
# rec = KaldiRecognizer(rec_model, SAMPLE_RATE)
# rec.SetWords(True)

q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, flush=True)
    q.put(bytes(indata))

# def recognize():
#     print("Listening... speak into the microphone.")
#     last_word_time = time.time() + 5
#     result_text = ""
#
#     with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype='int16',
#                            channels=1, callback=audio_callback):
#         while True:
#             data = q.get()
#
#             if rec.AcceptWaveform(data):
#                 res = json.loads(rec.Result())
#                 if res.get("text"):
#                     last_word_time = time.time()  # speech detected
#                     result_text += " " + res["text"]
#                     print("Final chunk:", res["text"])
#             else:
#                 partial = json.loads(rec.PartialResult()).get("partial", "")
#                 if partial:
#                     last_word_time = time.time()
#                     print("Partial:", partial)
#
#             if time.time() - last_word_time > NO_WORDS_DURATION:
#                 print("\nFinal result:", result_text.strip())
#                 return result_text.strip()

def keyboard_text():
    return input("Start chatting now!")


import ollama
import threading

emotion_queue = queue.Queue()

# chat_history = 'Complete the dialogue on behalf of the female streamer, keep the dialogue to be only 1 sentence and very short. Based on your emotions, add "/" and one of the following emotions at the start of your output when talking to the audience: angry, cry, sad, serious, smile, think. Do not put any other punctuation around the final part, (example: "smile/ Hi, how are you?"). Audience: '

chat_history = ("You are a streamer that is funny and quirky. Interact with the audience, and based on your emotions at the end of your response put a '/' followed by one of the follwing: think, serious, smile, anger, shame, cry, sad. Keep your responses short \n Audience: ")

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

def chatbot_loop():
    global chat_history
    while True:
        text = keyboard_text()
        chat_history += "" + text
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=chat_history
        )
        emotion = response.text.split("/", 1)[-1]
        output = response.text.split("/", 1)[0]
        chat_history += "\n Streamer: " + response.text.split("\n")[-1] + "\n Audience: "
        print(chat_history)
        try:
            emotion_queue.put_nowait(emotion)
        except queue.Full:
            pass
        play_streaming_tts(inference, output)

import pygame
import live2d.v2 as live2d
from live2d.v2.params import StandardParams
live2d.setLogEnable(False)

def s_call(group, no):
    print(group, no)

def f_call():
    print("end")

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

def text_to_image(text):
    img = PILImage.open('Transparent.png')
    img = img.resize((450, 700))
    imgDraw = ImageDraw.Draw(img)
    imgDraw.text((0, 500), text, font_size=15, fill=(255, 255, 255))
    img = img.transpose(PILImage.Transpose.FLIP_TOP_BOTTOM)
    img.save("Subtitle.png")

from live2d.utils.image import Image

def main():
    chat_thread = threading.Thread(target=chatbot_loop, daemon=True)
    chat_thread.start()

    pygame.init()
    live2d.init()

    display = (450, 700)
    pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)

    live2d.glewInit()

    model = live2d.LAppModel()

    text_to_image("Hello")

    subtitle = Image(
        os.path.join("subtitle.png")
    )

    current_text = ""

    if live2d.LIVE2D_VERSION == 3:
        model.LoadModelJson(os.path.join(
                                        ""
                                         ))
    elif live2d.LIVE2D_VERSION == 2:
        model.LoadModelJson(os.path.join("Resources/v2/anon/casual-2023/model.json"), disable_precision=True)
    model.Resize(*display)

    running = True

    # params = Params()
    # td.Thread(None, capture_task, "Capture Task", (params,0), daemon=True).start()

    model.SetAutoBreathEnable(True)
    model.SetAutoBlinkEnable(True)

    lipSyncN = 1.7

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.MOUSEBUTTONDOWN:
                print("set random expression")
                model.SetRandomExpression()

        try:
            current_rms = rms_queue.get_nowait()
            model.SetParameterValue(StandardParams.ParamMouthOpenY, current_rms * lipSyncN)

        except queue.Empty:
            pass

        try:
            current_emotion = emotion_queue.get_nowait()
            current_emotion = current_emotion.lower()
            print("Emotion: " + current_emotion)
            # if current_emotion in model.expressions.keys():
            #     model.SetExpression(current_emotion)
            if "think" in current_emotion:
                model.SetExpression("thinking01")
            elif "serious" in current_emotion:
                model.SetExpression("serious01")
            elif "smile" in current_emotion:
                model.SetExpression("smile01")
            elif "anger" in current_emotion:
                model.SetExpression("angry01")
            elif "shame" in current_emotion:
                model.SetExpression("shame01")
            elif "sad" in current_emotion:
                model.SetExpression("sad01")
            elif "cry" in current_emotion:
                model.SetExpression("cry01")

        except queue.Empty:
            pass

        try:
            new_text = text_queue.get_nowait()
            if current_text != new_text:
                current_text = new_text
                text_to_image(current_text)
                subtitle = Image(os.path.join("subtitle.png"))

        except queue.Empty:
            pass

        if not running:
            break

        model.SetParameterValue("Param14", 1, 1)

        live2d.clearBuffer()
        model.Update()
        model.Draw()
        subtitle.Draw()
        pygame.display.flip()
        pygame.time.wait(int(1000 / 60))

    live2d.dispose()
    pygame.quit()
    quit()

main()
