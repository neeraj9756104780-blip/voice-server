import os
import io
import wave

import numpy as np
from flask import Flask, request, Response
from google import genai
from google.genai import types

CHAT_MODEL = "gemini-3.8-flash"
TTS_MODEL = "gemini-3.8-flash-tts"
VOICE = "Kore"
IN_RATE = 8000

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
app = Flask(__name__)


def make_wav(pcm16, rate):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm16)
    return buf.getvalue()


@app.get("/")
def home():
    return "ok"


@app.post("/ask")
def ask():
    raw = np.frombuffer(request.data, dtype=np.uint8).astype(np.int16)
    pcm16 = ((raw - 128) * 256).astype("<i2").tobytes()
    wav = make_wav(pcm16, IN_RATE)

    r = client.models.generate_content(
        model=CHAT_MODEL,
        contents=[
            types.Part.from_bytes(data=wav, mime_type="audio/wav"),
            "Listen to this audio question and answer it in 1-2 short "
            "sentences, in the same language that was spoken. Plain text "
            "only, no emojis or symbols.",
        ],
    )
    answer = (r.text or "Sorry, I could not hear that.").strip()
    print("ANSWER:", answer)

    t = client.models.generate_content(
        model=TTS_MODEL,
        contents=answer,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=VOICE
                    )
                )
            ),
        ),
    )
    pcm24 = t.candidates[0].content.parts[0].inline_data.data
    a = np.frombuffer(pcm24, dtype="<i2").astype(np.int32)
    a = a[: len(a) // 2 * 2].reshape(-1, 2).mean(axis=1).astype("<i2")
    return Response(a.tobytes(), mimetype="application/octet-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
