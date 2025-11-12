import os
import torch
import pvporcupine
import pyaudio
import wave
import numpy as np
from dotenv import load_dotenv
import time
from vosk import Model, KaldiRecognizer
import json
from typing import Optional, Tuple, List, Callable


class VoiceCommandController:
    """
    A library class for voice-controlled drone swarm commands.
    Handles wake word detection, speech recording, transcription, and command parsing.
    """

    def __init__(
        self,
        porcupine_access_key: Optional[str] = None,
        wake_word_path: str = "models/Drone-Swarm_en_windows_v3_0_0.ppn",
        vosk_model_path: str = "models/vosk-model-small-en-us-0.15",
        audio_device_index: Optional[int] = 2,
        sample_rate: int = 16000
    ):
        """
        Initialize the Voice Command Controller.

        Args:
            porcupine_access_key: Porcupine API key (loads from .env if None)
            wake_word_path: Path to wake word model file
            vosk_model_path: Path to Vosk speech recognition model
            audio_device_index: PyAudio device index (None for default)
            sample_rate: Audio sample rate in Hz
        """
        # Load environment variables
        load_dotenv()
        self.access_key = porcupine_access_key or os.getenv("PORCUPINE_ACCESS_KEY")
        self.sample_rate = sample_rate
        self.audio_device_index = audio_device_index

        # Load Silero VAD
        print("Loading VAD model...")
        self.vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad"
        )
        self.get_speech_timestamps = utils[0]

        # Load Vosk model
        print("Loading Vosk model...")
        self.vosk_model = Model(vosk_model_path)
        print("Vosk model loaded")

        # Create Porcupine wake word detector
        self.porcupine = pvporcupine.create(
            access_key=self.access_key,
            keyword_paths=[wake_word_path]
        )

        # Initialize PyAudio
        self.pa = pyaudio.PyAudio()
        self.audio_stream = None
        self._setup_audio_stream()

        # Command callback
        self.command_callback: Optional[Callable] = None

    def _setup_audio_stream(self):
        """Initialize PyAudio stream"""
        self.audio_stream = self.pa.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            input_device_index=self.audio_device_index,
            frames_per_buffer=self.porcupine.frame_length
        )

    def record_until_silence(
        self,
        response: bool = False,
        max_duration: int = 10
    ) -> Optional[np.ndarray]:
        """
        Records from the microphone until silence is detected.

        Args:
            response: If True, uses shorter silence limit (for yes/no responses)
            max_duration: Maximum recording duration in seconds

        Returns:
            Audio data as numpy array, or None if no audio recorded
        """
        silence_limit = 2.3 if response else 1.5
        print("Listening for speech...")

        chunk_size = int(0.5 * self.sample_rate)  # 0.5s chunks
        buffer = []
        silence_start = None
        start_time = time.time()
        temp_chunk = np.array([], dtype=np.float32)

        while True:
            # Read audio data
            data = self.audio_stream.read(
                self.porcupine.frame_length,
                exception_on_overflow=False
            )
            pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            temp_chunk = np.concatenate((temp_chunk, pcm))

            # Check VAD when we have ~0.5s of audio
            if len(temp_chunk) >= chunk_size:
                audio_tensor = torch.tensor(temp_chunk).unsqueeze(0)
                speech = self.get_speech_timestamps(audio_tensor, self.vad_model)

                if len(speech) > 0:
                    buffer.append(temp_chunk)
                    silence_start = None
                    print("Speaking...", end="\r")
                else:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > silence_limit:
                        print("\nSilence detected. Stopping recording.")
                        break

                temp_chunk = np.array([], dtype=np.float32)

            if time.time() - start_time > max_duration:
                print("\n⏱Max duration reached. Stopping.")
                break

        if len(buffer) == 0:
            return None

        return np.concatenate(buffer)

    def save_wav(self, audio_data: np.ndarray, filename: str):
        """
        Save audio data as WAV file.

        Args:
            audio_data: Audio data as float32 numpy array
            filename: Output filename
        """
        audio_data = (audio_data * 32767).astype(np.int16)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())

    def transcribe_vosk(self, audio_path: str) -> str:
        """
        Transcribe audio file using Vosk model.

        Args:
            audio_path: Path to WAV file

        Returns:
            Transcribed text
        """
        if not os.path.exists(audio_path):
            return "ERROR: Audio file not found"

        try:
            wf = wave.open(audio_path, "rb")

            # Verify audio format
            if (wf.getnchannels() != 1 or
                wf.getsampwidth() != 2 or
                wf.getframerate() != 16000):
                wf.close()
                raise ValueError("Audio must be 16kHz, 16-bit, mono")

            # Get command vocabulary
            commands = self.get_supported_commands()
            rec = KaldiRecognizer(
                self.vosk_model,
                wf.getframerate(),
                json.dumps(commands)
            )
            rec.SetWords(True)

            results = []
            print("Transcribing command...")

            # Process audio in chunks
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break

                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    results.append(result)
                    print(".", end="", flush=True)

            # Get final result
            final_result = json.loads(rec.FinalResult())
            results.append(final_result)
            print()  # New line

            wf.close()

            # Combine all text fragments
            text = " ".join([r.get("text", "") for r in results if r.get("text")])
            return text.strip()

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return f"ERROR: {str(e)}"

    @staticmethod
    def get_supported_commands() -> List[str]:
        """
        Get list of all supported voice commands.

        Returns:
            List of command strings
        """
        return [
            # start
            "start one", "start two", "start three", "start four", "start",
            # scan
            "scan one", "scan two", "scan three", "scan four", "scan",
            # pause scan
            "pause scan one", "pause scan two", "pause scan three", "pause scan four", "pause scan",
            # resume scan
            "resume scan one", "resume scan two", "resume scan three", "resume scan four", "resume scan",
            # restart scan
            "restart scan one", "restart scan two", "restart scan three", "restart scan four", "restart scan",
            # mark
            "mark one", "mark two", "mark three", "mark four", "mark",
            # pause mark
            "pause mark one", "pause mark two", "pause mark three", "pause mark four", "pause mark",
            # resume mark
            "resume mark one", "resume mark two", "resume mark three", "resume mark four", "resume mark",
            # generate path
            "generate path",
            # start guidance
            "start guidance",
            # pause guidance
            "pause guidance",
            # user acceptance/rejection
            "yes", "no"
        ]

    def parse_command(self, input_text: str) -> Tuple[str, any]:
        """
        Parse natural language command into service call format.

        Args:
            input_text: Voice command text

        Returns:
            Tuple of (service_name, parameters)
        """
        input_text = input_text.lower().strip()
        parts = input_text.split()

        if not parts:
            return ["/unknown", {}]

        word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4}
        num = next((word_to_num[w] for w in parts if w in word_to_num), "ALL_DRONES")

        # Command mapping rules
        if parts[0] == "start" and "guidance" not in input_text:
            return ["/initialise", num]

        if parts[0] == "scan":
            return ["/generate_scan_waypoints", num]

        if len(parts) >= 2:
            if parts[:2] == ["pause", "scan"]:
                return ["/pause_drone", num]

            if parts[:2] == ["resume", "scan"]:
                return ["/resume_drone", num]

            if parts[:2] == ["restart", "scan"]:
                return ["/restart_scan", num]

            if parts[:2] == ["pause", "mark"]:
                return ["/mark_mines_pause", num]

            if parts[:2] == ["resume", "mark"]:
                return ["/mark_mines_resume", num]

            if parts[:2] == ["generate", "path"]:
                return ["/generate_path", {}]

            if parts[:2] == ["start", "guidance"]:
                return ["/start_guidance", {}]

            if parts[:2] == ["pause", "guidance"]:
                return ["/pause_guidance", {}]

        if parts[0] == "mark":
            return ["/mark_mines", num]

        return ["/unknown", {}]

    def set_command_callback(self, callback: Callable):
        """
        Set callback function to be called when command is executed.

        Args:
            callback: Function that takes (service, params, execute) as arguments
        """
        self.command_callback = callback

    def process_wake_word_detection(self) -> bool:
        """
        Process one frame of audio for wake word detection.

        Returns:
            True if wake word detected, False otherwise
        """
        pcm = self.audio_stream.read(
            self.porcupine.frame_length,
            exception_on_overflow=False
        )
        pcm = memoryview(pcm)
        pcm = [int.from_bytes(pcm[i:i + 2], byteorder="little", signed=True)
               for i in range(0, len(pcm), 2)]

        keyword_index = self.porcupine.process(pcm)
        return keyword_index >= 0

    def handle_voice_command(self) -> Optional[Tuple[str, any, bool]]:
        """
        Handle complete voice command flow: record, transcribe, confirm.

        Returns:
            Tuple of (service, params, execute) or None if no valid command
        """
        print("*BEEP/BUZZER* Speak your command now...")
        time.sleep(0.3)

        # Record command
        recorded_audio = self.record_until_silence(response=False)
        if recorded_audio is None:
            print("⚠️ No audio recorded")
            return None

        # Save and transcribe
        filename = "spoken_command.wav"
        self.save_wav(recorded_audio, filename)
        text_transcribed = self.transcribe_vosk(filename)

        if not text_transcribed:
            print("⚠️ No speech detected in recording")
            return None

        print(f"Transcription: '{text_transcribed}'")
        service, params = self.parse_command(text_transcribed)
        print(f"Service: {service}")
        print(f"Arguments: {params}")

        # Handle unknown command
        if service == "/unknown":
            print("Invalid command")
            time.sleep(3)
            return None

        # Get user confirmation
        print(f"Do I execute: {text_transcribed}, with service: {service}")
        response_audio = self.record_until_silence(response=True)

        if response_audio is None:
            print("⚠️ No response recorded")
            return None

        res_filename = "response_to_command.wav"
        self.save_wav(response_audio, res_filename)
        response_text = self.transcribe_vosk(res_filename)

        if response_text == "yes":
            print("Executing command demanded...")
            return (service, params, True)
        elif response_text == "no":
            print("Requested command denied!!")
            return (service, params, False)
        else:
            print("Invalid user response")
            time.sleep(3)
            return None

    def run(self, blocking: bool = True):
        """
        Run the main voice command loop.

        Args:
            blocking: If True, runs indefinitely until KeyboardInterrupt
        """
        print("🚀 Listening for wake word... (Ctrl+C to stop)")

        try:
            while True:
                if self.process_wake_word_detection():
                    print(f"Wake word detected!")
                    result = self.handle_voice_command()

                    if result and self.command_callback:
                        service, params, execute = result
                        self.command_callback(service, params, execute)

                    if not blocking:
                        break

        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.pa:
            self.pa.terminate()
        if self.porcupine:
            self.porcupine.delete()
        print("Cleanup complete")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()