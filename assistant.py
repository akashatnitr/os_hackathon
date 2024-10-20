import asyncio
from typing import Annotated
import face_recognition
import cv2
import os
import numpy as np
from datetime import datetime
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import ChatContext, ChatImage, ChatMessage
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero
schedule = """**Day 1 — Saturday Oct 19th** [times are US Pacific]

- 11:00 — Doors open
- 11:30 — Welcome and short sponsor intros
- 12:00 — Hacking! (Again, if you’re in SF, feel free to work from wherever)
- 1:00 — Lunch in SF
- 6:00 — Dinner in SF
- 8:00 — Doors close for the night in SF. (Discord never sleeps. :-))

Day 2 — Sunday Oct 20th

- 9:00 — Doors open and breakfast in SF
- 12:00 — Lunch in SF
- 9:00 - 4:00 — Hacking
- **4:00 — Project submissions deadline**
- 5:00 — Award presentations and demos
- 6:00 — Event wrap-up"""
KNOWN_FACES_PATH = "faces_known"
UNKNOWN_FACES_PATH = "faces_unknown"

face_encodings_db = []
face_names_db = []

class VisionAssistant(agents.llm.FunctionContext):
    @agents.llm.ai_callable(
        description=("Triggered when vision capabilities are required, such as analyzing an image or video feed.")
    )
    async def process_image(
        self,
        user_request: Annotated[
            str,
            agents.llm.TypeInfo(description="The user request that initiated this function"),
        ],
    ):
        print(f"Processing image request: {user_request}")
        return None


async def fetch_video_stream(room: rtc.Room):
    video_feed = asyncio.Future[rtc.RemoteVideoTrack]()
    for _, participant in room.remote_participants.items():
        for _, publication in participant.track_publications.items():
            if publication.track and isinstance(publication.track, rtc.RemoteVideoTrack):
                video_feed.set_result(publication.track)
                print(f"Selected video feed {publication.track.sid}")
                break
    return await video_feed


async def main_entry(ctx: JobContext):
    await ctx.connect()
    print(f"Connected to room: {ctx.room.name}")

    chat_session = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "Your name is Alfred. You are a digital receptionist at a hackathon."
                    "WiFi: OpenSource Hackathon, password: Attention is all you need. Men's restroom: back, Women's restroom: front. Kitchen: 3rd floor. "
                    f"Schedule is: {schedule}"
                    "You communicate using voice and vision. Provide concise answers, avoid emojis or unusual punctuation."
                ),
            )
        ]
    )

    gpt_model = openai.LLM(model="gpt-4o")
    voice_adapter = tts.StreamAdapter(
        tts=openai.TTS(voice="alloy"),
        sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
    )

    last_frame: rtc.VideoFrame | None = None

    assistant_instance = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=gpt_model,
        tts=voice_adapter,
        fnc_ctx=VisionAssistant(),
        chat_ctx=chat_session,
    )

    chat_handler = rtc.ChatManager(ctx.room)

    async def respond_to_user(response_text: str, include_image: bool = False):
        response_content: list[str | ChatImage] = [response_text]
        if include_image and last_frame:
            response_content.append(ChatImage(image=last_frame))

        chat_session.messages.append(ChatMessage(role="user", content=response_content))

        response_stream = gpt_model.chat(chat_ctx=chat_session)
        await assistant_instance.say(response_stream, allow_interruptions=True)

    @chat_handler.on("message_received")
    def handle_incoming_message(msg: rtc.ChatMessage):
        if msg.message:
            asyncio.create_task(respond_to_user(msg.message, include_image=False))

    @assistant_instance.on("function_calls_finished")
    def handle_function_completion(called_functions: list[agents.llm.CalledFunction]):
        if len(called_functions) == 0:
            return
        user_request = called_functions[0].call_info.arguments.get("user_request")
        if user_request:
            asyncio.create_task(respond_to_user(user_request, include_image=True))

    assistant_instance.start(ctx.room)

    await asyncio.sleep(1)
    await assistant_instance.say("Hi there! How can I assist?", allow_interruptions=True)

    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        video_feed = await fetch_video_stream(ctx.room)
        async for frame_event in rtc.VideoStream(video_feed):
            last_frame = frame_event.frame


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=main_entry))
