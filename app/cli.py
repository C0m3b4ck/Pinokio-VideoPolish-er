#!/usr/bin/env python3
"""
VideoPolish-er CLI Tool

A command-line tool for transcribing and cleaning up audio/video files
with automatic noise removal and stutter detection.
"""

import argparse
import os
import sys
import subprocess
import shlex
import time
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class Style:
    """ANSI escape codes for colored terminal output."""
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


def supports_color():
    """Check if terminal supports ANSI colors."""
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()


if not supports_color():
    for attr in dir(Style):
        if attr.isupper() and attr != "RESET":
            setattr(Style, attr, "")


def print_banner():
    """Print the application banner."""
    print(f"""
{Style.BRIGHT_CYAN}{Style.BOLD}
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   VideoPolish-er                                            ║
║   Transcription & Audio Cleanup Tool                        ║
║   Based on Transcribix                                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET}
""")


def print_status(message: str, color: str = Style.CYAN):
    """Print a status message."""
    print(f"{color}{message}{Style.RESET}")


def print_error(message: str):
    """Print an error message."""
    print(f"{Style.BRIGHT_RED}{Style.BOLD}Error: {message}{Style.RESET}", file=sys.stderr)


def print_success(message: str):
    """Print a success message."""
    print(f"{Style.BRIGHT_GREEN}{Style.BOLD}✓ {message}{Style.RESET}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"{Style.BRIGHT_YELLOW}{Style.BOLD}⚠ {message}{Style.RESET}")


def print_info(message: str):
    """Print an info message."""
    print(f"{Style.CYAN}{message}{Style.RESET}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VideoPolish-er: Transcription & Audio Cleanup Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input video.mp4 --remove-silence --remove-stutters
  %(prog)s --input podcast.wav --model faster-whisper --language en
  %(prog)s --input interview.mp4 --silence-threshold -35 --min-silence-duration 300
  %(prog)s --input video.mp4 --script original.srt --similarity-threshold 85
  %(prog)s --input video.mp4 --format ass --burn
        """
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input audio/video file path"
    )

    parser.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory (default: ./output)"
    )

    parser.add_argument(
        "--model", "-m",
        default="faster-whisper",
        choices=["faster-whisper", "whisperx", "stable-ts", "parakeet", "canary",
                 "distil-whisper", "moonshine", "sensevoice", "vosk", "whisper-original",
                 "whisper-cpp"],
        help="Transcription model (default: faster-whisper)"
    )

    parser.add_argument(
        "--language", "-l",
        default="en",
        help="Language code (default: en)"
    )

    parser.add_argument(
        "--model-size",
        default="large-v3",
        help="Model size (default: large-v3)"
    )

    parser.add_argument(
        "--remove-silence",
        action="store_true",
        help="Enable silence removal"
    )

    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=-40,
        help="Silence threshold in dB (default: -40)"
    )

    parser.add_argument(
        "--min-silence-duration",
        type=float,
        default=500,
        help="Minimum silence duration in ms (default: 500)"
    )

    parser.add_argument(
        "--remove-stutters",
        action="store_true",
        help="Enable stutter removal"
    )

    parser.add_argument(
        "--stutter-patterns",
        type=str,
        default="uhm,uh,um,er,ah,hmm,mhm,erm,umm",
        help="Comma-separated stutter patterns (default: uhm,uh,um,er,ah,hmm,mhm,erm,umm)"
    )

    parser.add_argument(
        "--format",
        choices=["srt", "vtt", "ass", "json"],
        default="srt",
        help="Output format (default: srt)"
    )

    parser.add_argument(
        "--words-per-chunk",
        type=int,
        default=3,
        help="Words per subtitle chunk (default: 3)"
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use (default: auto)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--script", "-s",
        type=str,
        help="Original script file for verification (SRT, VTT, ASS, TXT)"
    )

    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=80.0,
        help="Minimum similarity percentage to consider as valid (default: 80.0)"
    )

    parser.add_argument(
        "--burn",
        action="store_true",
        help="Burn subtitles into video using ffmpeg (requires ffmpeg installed)"
    )

    return parser.parse_args()


def validate_input_file(file_path: str) -> bool:
    """Validate that input file exists and is supported."""
    real_path = os.path.realpath(file_path)
    if not os.path.exists(real_path):
        print_error("File not found.")
        return False

    file_size_gb = os.path.getsize(real_path) / (1024**3)
    if file_size_gb > 10:
        print_error("File too large (>10GB).")
        return False

    supported_extensions = ['.mp4', '.wav', '.mp3', '.m4a', '.flac', '.ogg', '.mkv', '.avi', '.mov', '.webm']
    file_ext = os.path.splitext(real_path)[1].lower()

    if file_ext not in supported_extensions:
        print_error("Unsupported file format.")
        print_info(f"Supported formats: {', '.join(supported_extensions)}")
        return False

    return True


def create_output_directory(output_dir: str) -> bool:
    """Create output directory if it doesn't exist."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        return True
    except OSError as e:
        print_error(f"Failed to create output directory: {e}")
        return False


def validate_script_file(file_path: str) -> bool:
    """Validate that script file exists and is supported."""
    real_path = os.path.realpath(file_path)
    if not os.path.exists(real_path):
        print_error("Script file not found.")
        return False

    supported_extensions = ['.srt', '.vtt', '.ass', '.txt']
    file_ext = os.path.splitext(real_path)[1].lower()

    if file_ext not in supported_extensions:
        print_error("Unsupported script format.")
        print_info(f"Supported formats: {', '.join(supported_extensions)}")
        return False

    return True


def parse_srt_file(file_path: str) -> str:
    """Parse SRT file and extract text content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove SRT numbering and timestamps
    lines = content.strip().split('\n')
    text_lines = []

    for line in lines:
        line = line.strip()
        # Skip empty lines, numbers, and timestamp lines
        if not line:
            continue
        if line.isdigit():
            continue
        if '-->' in line:
            continue
        # Remove HTML tags if present
        clean_line = re.sub(r'<[^>]+>', '', line)
        text_lines.append(clean_line)

    return ' '.join(text_lines)


def parse_vtt_file(file_path: str) -> str:
    """Parse WebVTT file and extract text content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove VTT header and metadata
    lines = content.strip().split('\n')
    text_lines = []
    in_cues = False

    for line in lines:
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        # Skip WEBVTT header
        if line.startswith('WEBVTT'):
            continue
        # Skip NOTE blocks
        if line.startswith('NOTE'):
            continue
        # Check for timestamp lines
        if '-->' in line:
            in_cues = True
            continue
        # Skip cue identifiers (numbers or IDs)
        if in_cues and (line.isdigit() or re.match(r'^[a-zA-Z0-9_-]+$', line)):
            continue
        # Remove HTML tags if present
        clean_line = re.sub(r'<[^>]+>', '', line)
        if clean_line:
            text_lines.append(clean_line)

    return ' '.join(text_lines)


def parse_ass_file(file_path: str) -> str:
    """Parse ASS file and extract text content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.strip().split('\n')
    text_lines = []

    for line in lines:
        line = line.strip()
        # Look for Dialogue lines
        if line.startswith('Dialogue:'):
            # Extract text after the last comma
            parts = line.split(',')
            if len(parts) >= 10:
                text = ','.join(parts[9:])
                # Remove ASS override tags like {\b1}, {\i0}, etc.
                text = re.sub(r'\{[^}]+\}', '', text)
                text = text.strip()
                if text:
                    text_lines.append(text)

    return ' '.join(text_lines)


def parse_txt_file(file_path: str) -> str:
    """Parse plain text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content.strip()


def parse_script_file(file_path: str) -> str:
    """Parse script file based on extension and extract text."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.srt':
        return parse_srt_file(file_path)
    elif ext == '.vtt':
        return parse_vtt_file(file_path)
    elif ext == '.ass':
        return parse_ass_file(file_path)
    elif ext == '.txt':
        return parse_txt_file(file_path)
    else:
        print_error(f"Unsupported script format: {ext}")
        return ""


def extract_text_from_words(words: List[Dict]) -> str:
    """Extract plain text from word list."""
    return ' '.join(w.get("word", "").strip() for w in words)


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, remove extra spaces)."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using SequenceMatcher."""
    from difflib import SequenceMatcher

    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)

    if not norm1 and not norm2:
        return 100.0
    if not norm1 or not norm2:
        return 0.0

    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    return similarity * 100


def find_missing_words(original: str, processed: str) -> List[str]:
    """Find words in original that are missing from processed."""
    orig_words = set(normalize_text(original).split())
    proc_words = set(normalize_text(processed).split())

    missing = orig_words - proc_words
    return sorted(missing)


def verify_transcription(
    original_script: str,
    transcription: str,
    threshold: float = 80.0
) -> Tuple[bool, Dict]:
    """
    Verify transcription against original script.

    Returns:
        Tuple of (is_valid, stats_dict)
    """
    similarity = calculate_similarity(original_script, transcription)
    missing_words = find_missing_words(original_script, transcription)

    is_valid = similarity >= threshold

    stats = {
        "similarity": similarity,
        "threshold": threshold,
        "is_valid": is_valid,
        "missing_words": missing_words,
        "missing_count": len(missing_words),
        "original_length": len(original_script.split()),
        "transcription_length": len(transcription.split())
    }

    return is_valid, stats


def remove_silence_from_audio(
    audio_path: str,
    output_path: str,
    threshold_db: float = -40,
    min_silence_duration_ms: float = 500
) -> Tuple[bool, Dict]:
    """
    Remove silence from audio file.

    Returns:
        Tuple of (success, stats_dict)
    """
    try:
        import librosa
        import soundfile as sf
        import numpy as np
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent

        print_info("Loading audio file...")
        audio = AudioSegment.from_file(audio_path)

        # Detect non-silent chunks
        print_info(f"Detecting silence (threshold: {threshold_db}dB, min duration: {min_silence_duration_ms}ms)...")
        non_silent_ranges = detect_nonsilent(
            audio,
            min_silence_len=min_silence_duration_ms,
            silence_thresh=threshold_db
        )

        if not non_silent_ranges:
            print_warning("No speech detected in audio")
            return False, {"error": "No speech detected"}

        # Calculate stats
        total_duration = len(audio)
        speech_duration = sum(end - start for start, end in non_silent_ranges)
        silence_duration = total_duration - speech_duration
        silence_ratio = (silence_duration / total_duration) * 100 if total_duration > 0 else 0

        # Create cleaned audio
        print_info("Removing silence...")
        cleaned_audio = AudioSegment.empty()
        for start, end in non_silent_ranges:
            cleaned_audio += audio[start:end]

        # Export cleaned audio
        cleaned_audio.export(output_path, format="wav")

        stats = {
            "original_duration": total_duration / 1000,  # seconds
            "cleaned_duration": len(cleaned_audio) / 1000,  # seconds
            "silence_removed": silence_duration / 1000,  # seconds
            "silence_ratio": silence_ratio,
            "segments_detected": len(non_silent_ranges)
        }

        print_success(f"Silence removed: {stats['silence_removed']:.1f}s ({stats['silence_ratio']:.1f}%)")
        return True, stats

    except Exception as e:
        print_error("Failed to remove silence.")
        return False, {"error": str(e)}


def detect_and_remove_stutters(
    words: List[Dict],
    stutter_patterns: List[str]
) -> Tuple[List[Dict], Dict]:
    """
    Detect and remove stutters from transcribed words.

    Args:
        words: List of word dictionaries with 'word', 'start', 'end' keys
        stutter_patterns: List of stutter patterns to detect

    Returns:
        Tuple of (cleaned_words, stats_dict)
    """
    if not words:
        return words, {"stutters_removed": 0, "stutters_found": 0}

    # Create regex pattern for stutters (case-insensitive)
    stutter_regex = re.compile(
        r'\b(' + '|'.join(re.escape(p) for p in stutter_patterns) + r')\b',
        re.IGNORECASE
    )

    # Find stutters
    stutters_found = []
    cleaned_words = []

    for i, word in enumerate(words):
        word_text = word.get("word", "").strip()
        if stutter_regex.search(word_text):
            stutters_found.append({
                "index": i,
                "word": word_text,
                "start": word.get("start", 0),
                "end": word.get("end", 0)
            })
        else:
            cleaned_words.append(word)

    # Calculate stats
    stats = {
        "stutters_found": len(stutters_found),
        "stutters_removed": len(stutters_found),
        "stutter_words": [s["word"] for s in stutters_found],
        "original_word_count": len(words),
        "cleaned_word_count": len(cleaned_words)
    }

    if stutters_found:
        print_info(f"Found {len(stutters_found)} stutters: {', '.join(stats['stutter_words'][:10])}")
        if len(stats['stutter_words']) > 10:
            print_info(f"... and {len(stats['stutter_words']) - 10} more")
        print_success(f"Removed {len(stutters_found)} stutters")
    else:
        print_info("No stutters detected")

    return cleaned_words, stats


def transcribe_audio(
    audio_path: str,
    model_name: str = "faster-whisper",
    model_size: str = "large-v3",
    language: str = "en",
    device: str = "auto"
) -> List[Dict]:
    """
    Transcribe audio file using specified model.

    Returns:
        List of word dictionaries with 'word', 'start', 'end' keys
    """
    print_info(f"Transcribing with {model_name}...")

    try:
        if model_name == "faster-whisper":
            return transcribe_faster_whisper(audio_path, model_size, device, language)
        elif model_name == "whisperx":
            return transcribe_whisperx(audio_path, model_size, device, language)
        elif model_name == "stable-ts":
            return transcribe_stable_ts(audio_path, model_size, device, language)
        elif model_name == "parakeet":
            return transcribe_parakeet(audio_path)
        elif model_name == "canary":
            return transcribe_canary_qwen_with_alignment(audio_path, language)
        elif model_name == "distil-whisper":
            return transcribe_distil_whisper(audio_path, model_size, device, language)
        elif model_name == "moonshine":
            return transcribe_moonshine(audio_path)
        elif model_name == "sensevoice":
            return transcribe_sensevoice(audio_path)
        elif model_name == "vosk":
            return transcribe_vosk(audio_path)
        elif model_name == "whisper-original":
            return transcribe_whisper_original(audio_path, model_size, language)
        elif model_name == "whisper-cpp":
            return transcribe_whisper_cpp(audio_path, model_size, language)
        else:
            print_error(f"Unknown model: {model_name}")
            return []
    except Exception as e:
        print_error("Transcription failed.")
        return []


def transcribe_faster_whisper(
    audio_path: str,
    model_size: str = "large-v3",
    device: str = "auto",
    language: str = "en"
) -> List[Dict]:
    """Transcribe using faster-whisper."""
    from faster_whisper import WhisperModel

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        language=language,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200)
    )

    print_info(f"Language: {info.language} (prob={info.language_probability:.2f})")

    all_words = []
    for segment in segments:
        print_info(f"Segment: {segment.start:.1f}s → {segment.end:.1f}s")
        if segment.words:
            for w in segment.words:
                all_words.append({
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "probability": w.probability
                })

    return all_words


def transcribe_whisperx(
    audio_path: str,
    model_size: str = "large-v3",
    device: str = "auto",
    language: str = "en"
) -> List[Dict]:
    """Transcribe using whisperx."""
    import whisperx
    import torch

    device_type = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = "float16" if device_type == "cuda" else "int8"

    model = whisperx.load_model(model_size, device_type, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16, language=language)

    model_a, metadata = whisperx.load_align_model(language_code=language, device=device_type)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device_type)

    all_words = []
    for segment in result["segments"]:
        if "words" in segment:
            for w in segment["words"]:
                all_words.append({
                    "word": w["word"],
                    "start": w["start"],
                    "end": w["end"],
                    "score": w.get("score", 0)
                })

    return all_words


def transcribe_stable_ts(
    audio_path: str,
    model_size: str = "large-v3",
    device: str = "auto",
    language: str = "en"
) -> List[Dict]:
    """Transcribe using stable-ts."""
    import stable_whisper

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    model = stable_whisper.load_model(model_size, device=device)
    result = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        vad=True,
        regroup=True,
        only_voice_segments=True
    )

    all_words = []
    for seg in result.segments:
        if seg.words:
            for w in seg.words:
                all_words.append({"word": w.word, "start": w.start, "end": w.end})

    return all_words


def transcribe_parakeet(
    audio_path: str,
    model_name: str = "nvidia/parakeet-tdt-0.6b-v3"
) -> List[Dict]:
    """Transcribe using Parakeet."""
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    hypotheses = model.transcribe([audio_path], batch_size=1, return_hypotheses=True)

    all_words = []
    for hyp in hypotheses:
        if hasattr(hyp, "timestamp") and hyp.timestamp:
            for word, start, end in hyp.timestamp:
                all_words.append({
                    "word": word,
                    "start": start / 1000,
                    "end": end / 1000
                })

    return all_words


def transcribe_canary_qwen_with_alignment(
    audio_path: str,
    language: str = "en"
) -> List[Dict]:
    """Transcribe using Canary Qwen with WhisperX alignment."""
    import nemo.collections.asr as nemo_asr
    import whisperx
    import torch

    canary = nemo_asr.models.ASRModel.from_pretrained("nvidia/canary-qwen-2.5b")
    transcriptions = canary.transcribe([audio_path])
    best_text = transcriptions[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio = whisperx.load_audio(audio_path)
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)

    segments = [{"text": best_text, "start": 0, "end": len(audio) / 16000}]
    result = whisperx.align(segments, model_a, metadata, audio, device)

    all_words = []
    for segment in result["segments"]:
        if "words" in segment:
            for w in segment["words"]:
                all_words.append({
                    "word": w["word"],
                    "start": w["start"],
                    "end": w["end"]
                })

    return all_words


def transcribe_distil_whisper(
    audio_path: str,
    model_size: str = "distil-large-v3",
    device: str = "auto",
    language: str = "en"
) -> List[Dict]:
    """Transcribe using Distil-Whisper."""
    from faster_whisper import WhisperModel

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        language=language,
        vad_filter=True
    )

    all_words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                all_words.append({
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "probability": w.probability
                })

    return all_words


def transcribe_moonshine(
    audio_path: str,
    model_name: str = "moonshine/base"
) -> List[Dict]:
    """Transcribe using Moonshine."""
    import moonshine
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    import torch
    import soundfile as sf

    processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-base")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("UsefulSensors/moonshine-base")

    audio, sr = sf.read(audio_path)
    if sr != processor.feature_extractor.sampling_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=processor.feature_extractor.sampling_rate)

    inputs = processor(audio, return_tensors="pt", sampling_rate=processor.feature_extractor.sampling_rate)
    with torch.no_grad():
        outputs = model.generate(**inputs, return_timestamps=True)

    result = processor.batch_decode(outputs, return_timestamps=True)[0]

    all_words = []
    for text in result:
        for word in text.split():
            all_words.append({"word": word, "start": 0.0, "end": 0.5})

    return all_words


def transcribe_sensevoice(
    audio_path: str,
    model_name: str = "iic/SenseVoiceSmall",
    language: str = "auto"
) -> List[Dict]:
    """Transcribe using SenseVoice."""
    from funasr import AutoModel

    model = AutoModel(model=model_name, vad_model="fsmn-vad", disable_update=True)
    result = model.generate(input=audio_path, language=language, use_itn=True)

    text = result[0]["text"]
    clean_text = re.sub(r"<\|[^|]+\|>", "", text).strip()

    all_words = []
    for word in clean_text.split():
        all_words.append({"word": word, "start": 0.0, "end": 0.5})

    return all_words


def transcribe_vosk(
    audio_path: str,
    model_path: str = "vosk-model-small-en-us-0.15",
    sample_rate: int = 16000
) -> List[Dict]:
    """Transcribe using Vosk."""
    import vosk
    import soundfile as sf
    import numpy as np
    import json as _json

    model = vosk.Model(model_path)
    recognizer = vosk.KaldiRecognizer(model, sample_rate)
    recognizer.SetWords(True)

    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

    audio_int16 = (audio * 32767).astype(np.int16)
    chunk_size = 4000
    for i in range(0, len(audio_int16), chunk_size):
        chunk = audio_int16[i:i + chunk_size].tobytes()
        recognizer.AcceptWaveform(chunk)

    final_result = _json.loads(recognizer.FinalResult())

    all_words = []
    if "result" in final_result:
        for w in final_result["result"]:
            all_words.append({
                "word": w["word"],
                "start": w["start"],
                "end": w["end"],
                "confidence": w["conf"]
            })

    return all_words


def transcribe_whisper_original(
    audio_path: str,
    model_size: str = "large-v3",
    language: str = "en"
) -> List[Dict]:
    """Transcribe using original Whisper."""
    import whisper

    model = whisper.load_model(model_size)
    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        language=language
    )

    all_words = []
    for segment in result["segments"]:
        if "words" in segment:
            for w in segment["words"]:
                all_words.append({
                    "word": w["word"],
                    "start": w["start"],
                    "end": w["end"],
                    "probability": w.get("probability", 0)
                })

    return all_words


def transcribe_whisper_cpp(
    audio_path: str,
    model_size: str = "large-v3",
    language: str = "en"
) -> List[Dict]:
    """Transcribe using whisper.cpp."""
    from pywhispercpp.model import Model

    model = Model(model_size)
    segments = model.transcribe(
        audio_path,
        word_timestamps=True,
        language=language
    )

    all_words = []
    for segment in segments:
        if hasattr(segment, "words") and segment.words:
            for w in segment.words:
                all_words.append({
                    "word": w.word,
                    "start": w.t0 / 100,
                    "end": w.t1 / 100
                })

    return all_words


def words_to_srt(words: List[Dict], output_path: str, words_per_group: int = 3):
    """Convert word-level timestamps into SRT subtitles."""
    def fmt(seconds: float) -> str:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        ms = int(s % 1 * 1000)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"

    groups = []
    for i in range(0, len(words), words_per_group):
        chunk = words[i:i + words_per_group]
        text = " ".join(w["word"].strip() for w in chunk)
        groups.append((chunk[0]["start"], chunk[-1]["end"], text))

    lines = []
    for idx, (start, end, text) in enumerate(groups, 1):
        lines.extend([str(idx), f"{fmt(start)} --> {fmt(end)}", text, ""])

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print_success(f"SRT saved to {output_path} ({len(groups)} blocks)")


def words_to_vtt(words: List[Dict], output_path: str, words_per_group: int = 3):
    """Convert word-level timestamps into WebVTT subtitles."""
    def fmt(seconds: float) -> str:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        ms = int(s % 1 * 1000)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{ms:03d}"

    groups = []
    for i in range(0, len(words), words_per_group):
        chunk = words[i:i + words_per_group]
        text = " ".join(w["word"].strip() for w in chunk)
        groups.append((chunk[0]["start"], chunk[-1]["end"], text))

    lines = ["WEBVTT", ""]
    for idx, (start, end, text) in enumerate(groups, 1):
        lines.extend([str(idx), f"{fmt(start)} --> {fmt(end)}", text, ""])

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print_success(f"WebVTT saved to {output_path} ({len(groups)} blocks)")


def words_to_ass(words: List[Dict], output_path: str, words_per_group: int = 3):
    """Convert word-level timestamps into ASS subtitles."""
    def fmt(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h}:{m:02d}:{s:05.2f}"

    ass_content = """[Script Info]
Title: VideoPolish-er Subtitles
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1, 2,20,20,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    groups = []
    for i in range(0, len(words), words_per_group):
        chunk = words[i:i + words_per_group]
        text = " ".join(w["word"].strip() for w in chunk)
        groups.append((chunk[0]["start"], chunk[-1]["end"], text))

    for start, end, text in groups:
        start_time = fmt(start)
        end_time = fmt(end)
        text = text.replace("\\", "\\\\")
        text = text.replace("{", "\\{").replace("}", "\\}")
        ass_content += f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n"

    Path(output_path).write_text(ass_content, encoding="utf-8")
    print_success(f"ASS saved to {output_path} ({len(groups)} blocks)")


def words_to_json(words: List[Dict], output_path: str, words_per_group: int = 3):
    """Convert word-level timestamps into JSON format."""
    groups = []
    for i in range(0, len(words), words_per_group):
        chunk = words[i:i + words_per_group]
        text = " ".join(w["word"].strip() for w in chunk)
        groups.append({
            "text": text,
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
            "words": [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in chunk]
        })

    output = {
        "transcription": {
            "total_words": len(words),
            "total_groups": len(groups),
            "words_per_group": words_per_group
        },
        "groups": groups
    }

    Path(output_path).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print_success(f"JSON saved to {output_path} ({len(groups)} groups)")


def burn_subtitles_to_video(
    video_path: str,
    subtitle_path: str,
    output_path: str,
):
    """Burn subtitles onto video using ffmpeg."""
    if not os.path.isfile(video_path):
        print_error(f"Video file not found: {video_path}")
        return False
    if not os.path.isfile(subtitle_path):
        print_error(f"Subtitle file not found: {subtitle_path}")
        return False

    sub_path_quoted = shlex.quote(subtitle_path)
    filter_str = f"subtitles={sub_path_quoted}"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", filter_str,
        "-c:a", "copy",
        output_path,
    ]

    print_info("\nBurning subtitles into video...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print_error("ffmpeg timed out after 600 seconds")
        return False
    except FileNotFoundError:
        print_error("ffmpeg not found. Please install ffmpeg.")
        return False

    if result.returncode != 0:
        print_error("ffmpeg failed. Check that ffmpeg is installed and the video file is valid.")
        return False

    print_success(f"Video with burned subtitles saved to {output_path}")
    return True


def main():
    """Main function."""
    args = parse_arguments()

    # Print banner
    print_banner()

    # Validate input file
    if not validate_input_file(args.input):
        sys.exit(1)

    # Create output directory
    if not create_output_directory(args.output):
        sys.exit(1)

    # Parse stutter patterns
    stutter_patterns = [p.strip() for p in args.stutter_patterns.split(",") if p.strip()]

    # Track processing stats
    stats = {
        "input_file": args.input,
        "model": args.model,
        "language": args.language,
        "remove_silence": args.remove_silence,
        "remove_stutters": args.remove_stutters,
        "silence_stats": None,
        "stutter_stats": None,
        "transcription_stats": None,
        "verification_stats": None,
        "processing_time": 0
    }

    start_time = time.time()

    # Process audio file
    working_file = args.input

    # Step 1: Remove silence if requested
    if args.remove_silence:
        print_info("\n=== Step 1: Removing Silence ===")
        silence_output = os.path.join(args.output, "audio_no_silence.wav")
        success, silence_stats = remove_silence_from_audio(
            args.input,
            silence_output,
            args.silence_threshold,
            args.min_silence_duration
        )

        if success:
            working_file = silence_output
            stats["silence_stats"] = silence_stats
        else:
            print_warning("Continuing with original audio")
    else:
        print_info("\n=== Step 1: Skipping Silence Removal ===")

    # Step 2: Transcribe audio
    print_info("\n=== Step 2: Transcribing Audio ===")
    words = transcribe_audio(
        working_file,
        args.model,
        args.model_size,
        args.language,
        args.device
    )

    if not words:
        print_error("No words detected in audio")
        sys.exit(1)

    stats["transcription_stats"] = {
        "words_detected": len(words),
        "model_used": args.model
    }

    # Step 3: Remove stutters if requested
    if args.remove_stutters:
        print_info("\n=== Step 3: Removing Stutters ===")
        words, stutter_stats = detect_and_remove_stutters(words, stutter_patterns)
        stats["stutter_stats"] = stutter_stats
    else:
        print_info("\n=== Step 3: Skipping Stutter Removal ===")

    # Step 4: Generate output files
    print_info("\n=== Step 4: Generating Output Files ===")

    base_name = Path(args.input).stem
    output_files = []

    if args.format in ["srt", "all"]:
        srt_path = os.path.join(args.output, f"{base_name}.srt")
        words_to_srt(words, srt_path, args.words_per_chunk)
        output_files.append(srt_path)

    if args.format in ["vtt", "all"]:
        vtt_path = os.path.join(args.output, f"{base_name}.vtt")
        words_to_vtt(words, vtt_path, args.words_per_chunk)
        output_files.append(vtt_path)

    if args.format in ["ass", "all"]:
        ass_path = os.path.join(args.output, f"{base_name}.ass")
        words_to_ass(words, ass_path, args.words_per_chunk)
        output_files.append(ass_path)

    if args.format in ["json", "all"]:
        json_path = os.path.join(args.output, f"{base_name}.json")
        words_to_json(words, json_path, args.words_per_chunk)
        output_files.append(json_path)

    # Step 5: Burn subtitles into video if requested
    if args.burn:
        print_info("\n=== Step 5: Burning Subtitles Into Video ===")
        burn_sub_path = None
        if args.format in ["ass", "all"]:
            burn_sub_path = os.path.join(args.output, f"{base_name}.ass")
        elif args.format in ["srt", "all"]:
            burn_sub_path = os.path.join(args.output, f"{base_name}.srt")
        elif args.format in ["vtt", "all"]:
            burn_sub_path = os.path.join(args.output, f"{base_name}.vtt")
        elif args.format in ["json", "all"]:
            print_warning("JSON format cannot be burned into video, skipping burn step")
            burn_sub_path = None

        if burn_sub_path:
            video_ext = Path(args.input).suffix
            burn_output = os.path.join(args.output, f"{base_name}_burned{video_ext}")
            success = burn_subtitles_to_video(args.input, burn_sub_path, burn_output)
            if success:
                output_files.append(burn_output)
            else:
                print_warning("Subtitle burning failed, continuing with remaining steps")

    # Step 6: Verify against original script if provided
    if args.script:
        print_info("\n=== Step 6: Verifying Against Original Script ===")

        if not validate_script_file(args.script):
            print_warning("Skipping verification due to invalid script file")
        else:
            # Parse original script
            original_text = parse_script_file(args.script)
            if original_text:
                # Extract transcription text
                transcription_text = extract_text_from_words(words)

                # Verify
                is_valid, verification_stats = verify_transcription(
                    original_text,
                    transcription_text,
                    args.similarity_threshold
                )

                stats["verification_stats"] = verification_stats

                # Print verification results
                print_info(f"Similarity: {verification_stats['similarity']:.1f}% "
                          f"(threshold: {verification_stats['threshold']:.1f}%)")

                if is_valid:
                    print_success("Verification PASSED - Transcription is consistent with original script")
                else:
                    print_warning("Verification FAILED - Possible content loss detected")

                if verification_stats["missing_words"]:
                    print_warning(f"Missing words: {', '.join(verification_stats['missing_words'][:20])}")
                    if len(verification_stats["missing_words"]) > 20:
                        print_info(f"  ... and {len(verification_stats['missing_words']) - 20} more")

                print_info(f"Original script words: {verification_stats['original_length']}")
                print_info(f"Transcription words: {verification_stats['transcription_length']}")
            else:
                print_warning("Could not parse script file")

    # Calculate final stats
    end_time = time.time()
    stats["processing_time"] = end_time - start_time

    # Print summary
    print_info("\n" + "=" * 60)
    print_info("PROCESSING COMPLETE")
    print_info("=" * 60)

    print_info(f"Input: {args.input}")
    print_info(f"Model: {args.model}")
    print_info(f"Language: {args.language}")

    if stats["silence_stats"]:
        print_info(f"Silence removed: {stats['silence_stats']['silence_removed']:.1f}s "
                  f"({stats['silence_stats']['silence_ratio']:.1f}%)")

    if stats["stutter_stats"]:
        print_info(f"Stutters removed: {stats['stutter_stats']['stutters_removed']}")

    if stats["verification_stats"]:
        print_info(f"Verification: {'PASSED' if stats['verification_stats']['is_valid'] else 'FAILED'} "
                  f"({stats['verification_stats']['similarity']:.1f}%)")

    print_info(f"Words detected: {len(words)}")
    print_info(f"Processing time: {stats['processing_time']:.1f}s")

    if output_files:
        print_info("\nOutput files:")
        for f in output_files:
            print_success(f"  {f}")

    # Save stats to JSON
    stats_path = os.path.join(args.output, f"{base_name}_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print_info(f"\nStats saved to {stats_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())