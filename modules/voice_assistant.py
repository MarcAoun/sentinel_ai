# ============================================
# Sentinel AI - Voice Assistant Module
# OpenAI Realtime API (websocket, mic input, text output)
# ============================================

import asyncio
import base64
import json
import logging
import re
import threading
import time
from io import BytesIO
from urllib.request import Request, urlopen
import numpy as np
import sounddevice as sd
import websockets
from PIL import Image as PILImage
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger("sentinel.voice")

REALTIME_API_URL = "wss://api.openai.com/v1/realtime"
SAMPLE_RATE = 24000  # OpenAI Realtime API expects 24kHz
CHUNK_DURATION = 0.1  # Send audio in 100ms chunks
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)


class VoiceAssistant:
    """Voice assistant using OpenAI Realtime API.

    Captures mic audio, streams to the API, and plays back AI audio
    responses through the speaker.
    """

    def __init__(self, on_transcript=None, on_user_transcript=None, on_status=None,
                 on_enroll=None, on_remove=None, on_rename=None):
        """
        Args:
            on_transcript: callback(text) called when the AI produces response text
            on_user_transcript: callback(text) called when the user's speech is transcribed
            on_status: callback(status_str) called on state changes
            on_enroll: callback(name) -> str, called when AI wants to register a person
            on_remove: callback(name) -> str, called when AI wants to remove a person
            on_rename: callback(old_name, new_name) -> str, called when AI wants to rename a person
        """
        self.on_transcript = on_transcript or (lambda t: None)
        self.on_user_transcript = on_user_transcript or (lambda t: None)
        self.on_status = on_status or (lambda s: None)
        self.on_enroll = on_enroll or (lambda name: "Enrollment not available")
        self.on_remove = on_remove or (lambda name: "Remove not available")
        self.on_rename = on_rename or (lambda old, new: "Rename not available")

        self._ws = None
        self._loop = None
        self._thread = None
        self._running = False
        self._session_active = False

        # Context from the camera/detection system
        self._context = {
            "person": "unknown",
            "name": "unidentified",
            "package": "no",
        }

        self._mic_stream = None
        self._speaker_stream = None
        self._response_text = ""
        self._volume = 0.8  # 0.0 to 1.0
        self._mic_muted = False
        self._ai_speaking = False       # True while AI audio is playing
        self._ai_speech_end = 0.0       # timestamp when AI stopped speaking
        self._echo_level = 0.0          # tracked mic level while AI speaks (speaker bleed)

        # Vision analysis state
        self._last_vision_time = 0
        self._last_vision_person = None
        self._visual_description = ""

        # Silence timeout (mutable so it can be reset from outside)
        self._silence_deadline = 0

        # Auto-enroll: track when we're talking to an unknown person
        self._unknown_greeting_active = False
        self._enrolled_this_session = False  # prevent double enrollment
        self._enrolled_name = None           # actual name enrolled (with corrections)

    # ------------------------------------------------------------------
    # Public API (thread-safe, called from main/Qt thread)
    # ------------------------------------------------------------------

    def start_session(self):
        """Start a voice conversation session in a background thread."""
        if self._session_active:
            logger.info("Session already active")
            return

        if not config.OPENAI_API_KEY:
            self.on_status("No API key — set OPENAI_API_KEY env variable")
            logger.error("OPENAI_API_KEY not set")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop_session(self):
        """Stop the current voice session."""
        self._running = False
        # Close the websocket to unblock recv/send
        if self._ws and self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._close_ws(), self._loop)

    async def _close_ws(self):
        """Close websocket from within the event loop."""
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

    def update_context(self, person="unknown", name="unidentified", package="no"):
        """Update the camera context fed to the AI."""
        self._context = {
            "person": person,
            "name": name,
            "package": package,
        }

    def refresh_persona(self):
        """Re-send session instructions with the current persona. Call from Qt thread."""
        if not self._session_active or not self._loop or not self._loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(self._push_instructions(), self._loop)

    async def _push_instructions(self):
        """Push updated system instructions to the active session."""
        if not self._ws:
            return
        context_str = (
            f"Person: {self._context['person']}\n"
            f"Name: {self._context['name']}\n"
            f"Package: {self._context['package']}"
        )
        if self._visual_description:
            context_str += f"\nWhat you see: {self._visual_description}"
        persona_prompt = config.VOICE_PERSONAS.get(config.VOICE_PERSONA, config.VOICE_PERSONAS["default"])
        instructions = f"{persona_prompt}\n{config._VOICE_RULES}\n\nCurrent context:\n{context_str}"
        await self._ws.send(json.dumps({
            "type": "session.update",
            "session": {"instructions": instructions},
        }))
        logger.info(f"Persona refreshed: {config.VOICE_PERSONA}")

    def reset_silence_timeout(self):
        """Reset the silence watchdog timer (call while face is visible)."""
        self._silence_deadline = time.time() + config.VOICE_SILENCE_TIMEOUT

    def trigger_greeting(self, person_type, name):
        """Make Sentinel speak first — greet a person proactively.

        Args:
            person_type: "known" or "unknown"
            name: person's name (or "unidentified")
        """
        if not self._session_active or not self._loop or not self._loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(
            self._send_greeting(person_type, name), self._loop
        )

    def update_visual_context(self, frame, person_name="unknown"):
        """Send a camera frame for vision analysis (called from Qt thread).

        Respects cooldown — won't re-analyze the same person within
        VOICE_VISION_COOLDOWN seconds.
        """
        if not self._session_active or not self._loop or not self._loop.is_running():
            return

        now = time.time()
        # Cooldown: skip if same person and within cooldown period
        if (person_name == self._last_vision_person
                and now - self._last_vision_time < config.VOICE_VISION_COOLDOWN):
            return

        self._last_vision_time = now
        self._last_vision_person = person_name

        # Schedule analysis on the voice assistant's event loop
        asyncio.run_coroutine_threadsafe(
            self._analyze_and_update(frame.copy(), person_name), self._loop
        )

    @property
    def is_active(self):
        return self._session_active

    def set_volume(self, level):
        """Set speaker volume (0.0 to 2.0)."""
        self._volume = max(0.0, min(2.0, level))

    def mute_mic(self):
        """Mute the microphone (stop sending audio to API)."""
        self._mic_muted = True
        logger.info("Mic muted")

    def unmute_mic(self):
        """Unmute the microphone (resume sending audio to API)."""
        self._mic_muted = False
        logger.info("Mic unmuted")

    @property
    def is_mic_muted(self):
        return self._mic_muted

    # ------------------------------------------------------------------
    # Internal — async event loop in background thread
    # ------------------------------------------------------------------

    def _run_loop(self):
        """Entry point for the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._session())
        except Exception as e:
            logger.error(f"Voice session error: {e}")
            self.on_status(f"Error: {e}")
        finally:
            self._session_active = False
            self._running = False
            self.on_status("Session ended")
            logger.info("Voice session ended")

    async def _session(self):
        """Full lifecycle of one voice conversation."""
        self.on_status("Connecting...")
        logger.info("Connecting to OpenAI Realtime API")

        url = f"{REALTIME_API_URL}?model={config.VOICE_MODEL}"
        headers = {
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }

        try:
            async with websockets.connect(url, additional_headers=headers) as ws:
                self._ws = ws
                self._session_active = True
                logger.info("Connected to Realtime API")

                # Configure the session and wait for confirmation
                await self._configure_session()

                # Wait for session.updated confirmation before signaling ready
                try:
                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        msg = json.loads(raw)
                        if msg.get("type") == "session.updated":
                            logger.info("Session configuration confirmed by server")
                            break
                        elif msg.get("type") == "session.created":
                            logger.info("Session created by server")
                        elif msg.get("type") == "error":
                            logger.error(f"Session config error: {msg}")
                            break
                except asyncio.TimeoutError:
                    logger.warning("Timed out waiting for session.updated, continuing anyway")

                self.on_status("Connected — listening")

                # Start speaker output stream
                self._start_speaker()

                # Start mic capture and message receiver concurrently
                self._silence_deadline = time.time() + config.VOICE_SILENCE_TIMEOUT

                mic_task = asyncio.create_task(self._stream_mic(ws))
                recv_task = asyncio.create_task(self._receive_messages(ws))
                watchdog = asyncio.create_task(self._silence_watchdog())

                done, pending = await asyncio.wait(
                    [mic_task, recv_task, watchdog],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()

        except websockets.exceptions.InvalidStatusCode as e:
            msg = f"API connection failed: {e.status_code}"
            if e.status_code == 401:
                msg = "Invalid API key"
            elif e.status_code == 429:
                msg = "Rate limited — try again later"
            logger.error(msg)
            self.on_status(msg)
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.on_status(f"Connection error: {e}")
        finally:
            self._stop_speaker()
            self._ws = None
            self._session_active = False

    async def _configure_session(self):
        """Send session.update to configure the realtime session."""
        context_str = (
            f"Person: {self._context['person']}\n"
            f"Name: {self._context['name']}\n"
            f"Package: {self._context['package']}"
        )
        persona_prompt = config.VOICE_PERSONAS.get(config.VOICE_PERSONA, config.VOICE_PERSONAS["default"])
        instructions = f"{persona_prompt}\n{config._VOICE_RULES}\n\nCurrent context:\n{context_str}"

        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": instructions,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "gpt-4o-mini-transcribe",
                },
                "voice": config.VOICE_VOICE,
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
                "tools": [
                    {
                        "type": "function",
                        "name": "register_person",
                        "description": (
                            "Register and save a new person's face when they tell you their name. "
                            "Only call this when an UNKNOWN person clearly states their name. "
                            "IMPORTANT: Apply spelling corrections before calling — "
                            "e.g. 'Mark with a C' means name='Marc', 'Jon without an H' means name='Jon'."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The person's first name with any spelling corrections applied",
                                }
                            },
                            "required": ["name"],
                        },
                    },
                    {
                        "type": "function",
                        "name": "remove_person",
                        "description": (
                            "Remove a person's face from the system. Use when someone asks "
                            "to be removed or to delete their face. If they say 'remove me', "
                            "use the current person's name from context."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The person's name to remove",
                                }
                            },
                            "required": ["name"],
                        },
                    },
                    {
                        "type": "function",
                        "name": "update_person_name",
                        "description": (
                            "Update or correct a person's name spelling. Use when someone says "
                            "their name is spelled differently (e.g. 'it's with a C', "
                            "'actually it's Marc not Mark', 'change my name to...'). "
                            "Apply the spelling correction yourself before calling."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "old_name": {
                                    "type": "string",
                                    "description": "The current name stored in the system",
                                },
                                "new_name": {
                                    "type": "string",
                                    "description": "The corrected/new name, correctly spelled",
                                },
                            },
                            "required": ["old_name", "new_name"],
                        },
                    },
                ],
            },
        }
        await self._ws.send(json.dumps(session_config))
        logger.info("Session configured")

    # ------------------------------------------------------------------
    # Vision analysis (camera → gpt-4o-mini → session context)
    # ------------------------------------------------------------------

    async def _analyze_and_update(self, frame, person_name):
        """Analyze frame with gpt-4o-mini and update session instructions."""
        try:
            description = await self._analyze_image(frame, person_name)
            if description and self._ws:
                self._visual_description = description
                await self._update_session_with_vision()
                logger.info(f"Vision context updated: {description[:80]}...")
        except Exception as e:
            logger.error(f"Vision analysis error: {e}")

    async def _analyze_image(self, frame, person_name):
        """Call gpt-4o-mini chat completions to describe the camera frame."""
        loop = asyncio.get_event_loop()

        def _prepare_and_call():
            # Resize to 320x240 for speed/cost
            pil_img = PILImage.fromarray(frame)
            pil_img = pil_img.resize((320, 240), PILImage.LANCZOS)

            # JPEG encode → base64
            buf = BytesIO()
            pil_img.save(buf, format="JPEG", quality=60)
            b64_img = base64.b64encode(buf.getvalue()).decode("ascii")

            prompt = (
                "Briefly describe this person at the door in 1-2 sentences. "
                "Focus on: what they look like, what they're wearing, "
                "what they're carrying or holding, their expression/mood. "
                "Be casual and observational, like telling a friend. "
                "Do NOT start with 'I see' or 'The image shows'."
            )
            if person_name and person_name not in ("unknown", "unidentified"):
                prompt += f"\nThis person's name is {person_name}."

            payload = json.dumps({
                "model": config.VOICE_VISION_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_img}",
                                "detail": "low",
                            },
                        },
                    ],
                }],
                "max_tokens": 150,
            }).encode("utf-8")

            req = Request(
                "https://api.openai.com/v1/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            resp = urlopen(req, timeout=10)
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"].strip()

        return await loop.run_in_executor(None, _prepare_and_call)

    async def _update_session_with_vision(self):
        """Re-send session.update with visual description injected."""
        context_str = (
            f"Person: {self._context['person']}\n"
            f"Name: {self._context['name']}\n"
            f"Package: {self._context['package']}\n"
            f"What you see: {self._visual_description}"
        )
        persona_prompt = config.VOICE_PERSONAS.get(config.VOICE_PERSONA, config.VOICE_PERSONAS["default"])
        instructions = f"{persona_prompt}\n{config._VOICE_RULES}\n\nCurrent context:\n{context_str}"

        session_update = {
            "type": "session.update",
            "session": {
                "instructions": instructions,
            },
        }
        await self._ws.send(json.dumps(session_update))

    async def _extract_name(self, text):
        """Extract a person's name from speech using GPT-4o-mini."""
        loop = asyncio.get_event_loop()

        def _call_llm():
            try:
                payload = json.dumps({
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Extract the person's first name from what they said. "
                                "Apply any spelling corrections they mention "
                                "(e.g. 'Mark with a C' = Marc, 'Jon without an H' = Jon, "
                                "'S-A-R-A-H' = Sarah). "
                                "Return ONLY the correctly spelled first name, nothing else. "
                                "If no name is found, return NONE."
                            ),
                        },
                        {"role": "user", "content": text},
                    ],
                    "max_tokens": 20,
                    "temperature": 0,
                }).encode("utf-8")

                req = Request(
                    "https://api.openai.com/v1/chat/completions",
                    data=payload,
                    headers={
                        "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                resp = urlopen(req, timeout=5)
                result = json.loads(resp.read().decode("utf-8"))
                name = result["choices"][0]["message"]["content"].strip().strip(".,!?\"'")
                if name.upper() == "NONE" or len(name) < 2 or not name.isalpha():
                    return None
                return name.capitalize()
            except Exception as e:
                logger.warning(f"LLM name extraction failed: {e}")
                return None

        return await loop.run_in_executor(None, _call_llm)

    async def _try_auto_enroll(self, user_text):
        """Background task: extract name from speech and enroll if found."""
        try:
            extracted = await self._extract_name(user_text)
            if not extracted:
                return

            if self._enrolled_this_session:
                # AI already enrolled — but check if spelling needs correction
                # (e.g. AI registered "Mark" but user said "Mark with a C" → "Marc")
                if (self._enrolled_name
                        and extracted.lower() != self._enrolled_name.lower()):
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, self.on_rename, self._enrolled_name, extracted
                    )
                    self._enrolled_name = extracted
                    logger.info(f"Auto-corrected name: {self._enrolled_name} -> {extracted}, result={result}")
                return

            self._enrolled_this_session = True
            self._enrolled_name = extracted
            self._unknown_greeting_active = False
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.on_enroll, extracted
            )
            logger.info(f"Auto-enroll from speech: name={extracted}, result={result}")
        except Exception as e:
            logger.error(f"Auto-enroll error: {e}")

    @staticmethod
    def _is_remove_request(text):
        """Check if the user is asking to be removed/unenrolled."""
        text_lower = text.lower()
        remove_phrases = [
            "remove me", "unenroll me", "delete me", "forget me",
            "remove my face", "delete my face", "unenroll my face",
            "take me off", "remove my name", "delete my name",
            "unregister me", "deregister me",
        ]
        return any(phrase in text_lower for phrase in remove_phrases)

    async def _try_auto_remove(self):
        """Remove the current known person when they ask to be unenrolled."""
        try:
            name = self._context.get("name", "")
            if not name or name == "unidentified":
                logger.warning("Auto-remove: no name in context")
                return
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.on_remove, name
            )
            logger.info(f"Auto-remove from speech: name={name}, result={result}")
        except Exception as e:
            logger.error(f"Auto-remove error: {e}")

    async def _send_greeting(self, person_type, name):
        """Inject a hidden prompt to make Sentinel greet someone first."""
        if not self._ws:
            return

        if person_type == "known":
            prompt = f"[{name} just showed up at the door. Say hi — be yourself, keep it super casual.]"
        else:
            prompt = "[Someone you don't know just showed up. Say hi casually and ask who they are.]"
            self._unknown_greeting_active = True
            self._enrolled_this_session = False
            self._enrolled_name = None

        # Add a user-role text message to the conversation
        await self._ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        }))
        # Trigger the AI to respond
        await self._ws.send(json.dumps({"type": "response.create"}))
        logger.info(f"Greeting triggered for {person_type}: {name}")

    # ------------------------------------------------------------------
    # Speaker output
    # ------------------------------------------------------------------

    def _start_speaker(self):
        """Open a sounddevice OutputStream for audio playback."""
        try:
            self._speaker_stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                device=config.VOICE_SPEAKER_DEVICE,
            )
            self._speaker_stream.start()
            logger.info(f"Speaker stream started (device={config.VOICE_SPEAKER_DEVICE})")
        except Exception as e:
            logger.error(f"Failed to open speaker: {e}")
            self.on_status(f"Speaker error: {e}")
            self._speaker_stream = None

    def _stop_speaker(self):
        """Close the speaker output stream."""
        if self._speaker_stream:
            try:
                self._speaker_stream.stop()
                self._speaker_stream.close()
            except Exception:
                pass
            self._speaker_stream = None
            logger.info("Speaker stream stopped")

    def _play_audio(self, pcm16_bytes):
        """Write raw PCM16 bytes to the speaker, scaled by volume."""
        if not self._speaker_stream:
            return
        try:
            audio = np.frombuffer(pcm16_bytes, dtype=np.int16)
            audio = np.clip(
                audio.astype(np.float32) * self._volume, -32768, 32767
            ).astype(np.int16)
            self._speaker_stream.write(audio.reshape(-1, 1))
        except Exception as e:
            logger.warning(f"Speaker write error: {e}")

    # ------------------------------------------------------------------
    # Mic capture → API
    # ------------------------------------------------------------------

    async def _stream_mic(self, ws):
        """Capture audio from mic and stream to the API as pcm16 chunks."""
        loop = asyncio.get_event_loop()
        audio_queue = asyncio.Queue()

        def mic_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Mic status: {status}")
            pcm16 = (indata[:, 0] * 32767).astype(np.int16)
            loop.call_soon_threadsafe(audio_queue.put_nowait, pcm16.tobytes())

        device = config.VOICE_MIC_DEVICE
        try:
            self._mic_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=CHUNK_SAMPLES,
                device=device,
                callback=mic_callback,
            )
            self._mic_stream.start()
            logger.info(f"Mic stream started (device={device})")
        except Exception as e:
            logger.error(f"Failed to open mic: {e}")
            self.on_status(f"Mic error: {e}")
            return

        try:
            while self._running:
                try:
                    data = await asyncio.wait_for(audio_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                if self._mic_muted:
                    continue

                # Mute mic while AI is speaking to prevent echo/feedback
                if self._ai_speaking or (time.time() - self._ai_speech_end < 0.5):
                    continue

                b64_audio = base64.b64encode(data).decode("ascii")
                msg = {
                    "type": "input_audio_buffer.append",
                    "audio": b64_audio,
                }
                await ws.send(json.dumps(msg))
        finally:
            self._mic_stream.stop()
            self._mic_stream.close()
            self._mic_stream = None
            logger.info("Mic stream stopped")

    # ------------------------------------------------------------------
    # Receive messages from API
    # ------------------------------------------------------------------

    async def _receive_messages(self, ws):
        """Listen for API messages, extract text transcripts."""
        while self._running:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                logger.info("WebSocket closed")
                break

            msg = json.loads(raw)
            event_type = msg.get("type", "")

            if event_type == "session.created":
                logger.info("Session created by server")

            elif event_type == "response.audio.delta":
                # AI is speaking — suppress mic to prevent echo
                self._ai_speaking = True
                audio_b64 = msg.get("delta", "")
                if audio_b64:
                    pcm_data = base64.b64decode(audio_b64)
                    self._play_audio(pcm_data)

            elif event_type == "response.audio_transcript.delta":
                delta = msg.get("delta", "")
                self._response_text += delta

            elif event_type == "response.audio_transcript.done":
                transcript = msg.get("transcript", self._response_text)
                self._response_text = ""
                if transcript.strip():
                    self.on_transcript(transcript)
                    logger.info(f"Sentinel said: {transcript}")
                # Reset silence timer — conversation is active
                self._silence_deadline = time.time() + config.VOICE_SILENCE_TIMEOUT

            elif event_type == "conversation.item.input_audio_transcription.completed":
                user_text = msg.get("transcript", "").strip()
                if user_text:
                    logger.info(f"User said: {user_text}")
                    self.on_user_transcript(user_text)
                if (user_text
                        and self._unknown_greeting_active
                        and not self._enrolled_this_session):
                    asyncio.create_task(self._try_auto_enroll(user_text))
                # Detect unenroll requests from known persons
                if (user_text
                        and self._context.get("person") == "known"
                        and self._is_remove_request(user_text)):
                    asyncio.create_task(self._try_auto_remove())

            elif event_type == "input_audio_buffer.speech_started":
                self.on_status("Listening...")
                logger.debug("User started speaking")
                # Reset silence timer — someone is talking
                self._silence_deadline = time.time() + config.VOICE_SILENCE_TIMEOUT

            elif event_type == "input_audio_buffer.speech_stopped":
                self.on_status("Processing...")
                logger.debug("User stopped speaking")

            elif event_type == "response.done":
                # AI finished speaking — re-enable mic after brief echo buffer
                self._ai_speaking = False
                self._ai_speech_end = time.time()
                self._response_text = ""
                self.on_status("Connected — listening")
                # Reset silence timer after AI finishes responding
                self._silence_deadline = time.time() + config.VOICE_SILENCE_TIMEOUT

            elif event_type == "response.function_call_arguments.done":
                call_id = msg.get("call_id", "")
                func_name = msg.get("name", "")
                args_str = msg.get("arguments", "{}")

                result = None

                if func_name == "register_person":
                    try:
                        args = json.loads(args_str)
                        name = args.get("name", "").strip()
                        if name and not self._enrolled_this_session:
                            self._enrolled_this_session = True
                            self._enrolled_name = name
                            self._unknown_greeting_active = False
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(
                                None, self.on_enroll, name
                            )
                            # Push updated context so AI knows this person's name
                            await self._push_instructions()
                        elif self._enrolled_this_session:
                            # Auto-enroll may have used a corrected spelling
                            actual = self._enrolled_name or name
                            result = f"Successfully registered {actual}. You'll recognize them from now on."
                        else:
                            result = "No name provided"
                    except Exception as e:
                        result = f"Error: {e}"
                        logger.error(f"register_person error: {e}")

                elif func_name == "remove_person":
                    try:
                        args = json.loads(args_str)
                        name = args.get("name", "").strip()
                        if name:
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(
                                None, self.on_remove, name
                            )
                            # Push updated context (now unknown again)
                            await self._push_instructions()
                        else:
                            result = "No name provided"
                    except Exception as e:
                        result = f"Error: {e}"
                        logger.error(f"remove_person error: {e}")

                elif func_name == "update_person_name":
                    try:
                        args = json.loads(args_str)
                        old_name = args.get("old_name", "").strip()
                        new_name = args.get("new_name", "").strip()
                        if old_name and new_name:
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(
                                None, self.on_rename, old_name, new_name
                            )
                        else:
                            result = "Both old_name and new_name are required"
                    except Exception as e:
                        result = f"Error: {e}"
                        logger.error(f"update_person_name error: {e}")

                if result is not None:
                    await ws.send(json.dumps({
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": result,
                        },
                    }))
                    await ws.send(json.dumps({"type": "response.create"}))
                    logger.info(f"{func_name} called: args={args_str}, result={result}")

            elif event_type == "error":
                err = msg.get("error", {})
                logger.error(f"API error: {err.get('message', msg)}")
                self.on_status(f"API error: {err.get('message', 'unknown')}")

    async def _silence_watchdog(self):
        """Stop session after prolonged silence."""
        while self._running:
            await asyncio.sleep(2)
            if time.time() > self._silence_deadline:
                logger.info("Silence timeout — ending session")
                self.on_status("Silence timeout")
                self._running = False
                return

    # ------------------------------------------------------------------
    # Audio test utility
    # ------------------------------------------------------------------

    @staticmethod
    def test_mic(duration=3):
        """Record from mic and return True if audio was captured."""
        print(f"[MIC TEST] Recording {duration}s from mic...")
        try:
            audio = sd.rec(
                int(duration * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=config.VOICE_MIC_DEVICE,
            )
            sd.wait()
            peak = np.max(np.abs(audio))
            print(f"[MIC TEST] Done. Peak amplitude: {peak:.4f}")
            if peak < 0.001:
                print("[MIC TEST] WARNING: Very low audio — mic may not be working")
                return False
            print("[MIC TEST] Mic is working!")
            return True
        except Exception as e:
            print(f"[MIC TEST] Error: {e}")
            return False
