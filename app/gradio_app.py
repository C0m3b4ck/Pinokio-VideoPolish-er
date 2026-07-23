#!/usr/bin/env python3
"""
VideoPolish-er Gradio Web UI

A web interface for transcribing and cleaning up audio/video files
with automatic noise removal and stutter detection.
"""

import os
import re
import sys
import time
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Optional, List

import gradio as gr

from cli import (
    remove_silence_from_audio,
    detect_and_remove_stutters,
    transcribe_audio,
    verify_transcription,
    parse_script_file,
    extract_text_from_words,
    words_to_srt,
    words_to_vtt,
    words_to_ass,
    words_to_json,
    burn_subtitles_to_video,
    fix_grammar_punctuation,
    fix_text_with_llm,
)

SUPPORTED_AUDIO_EXTENSIONS = (".mp4", ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mkv", ".avi", ".mov", ".webm")
SUPPORTED_SCRIPT_EXTENSIONS = (".srt", ".vtt", ".ass", ".txt")
MODEL_CHOICES = [
    "faster-whisper", "whisperx", "stable-ts", "parakeet", "canary",
    "distil-whisper", "moonshine", "sensevoice", "vosk", "whisper-original",
    "whisper-cpp",
]
MODEL_SIZES = [
    "tiny", "base", "small", "medium", "large-v3",
    "distil-large-v3", "distil-small.en",
]
FORMAT_CHOICES = ["srt", "vtt", "ass", "json"]
DEVICE_CHOICES = ["auto", "cuda", "cpu"]


def sanitize_filename_stem(stem: str) -> str:
    """Remove characters from a filename stem that could cause path issues."""
    return re.sub(r'[^\w\-.]', '_', stem)[:128]


def validate_file(file_path: str, allowed_extensions: tuple) -> Optional[str]:
    """Return an error message if the file is invalid, else None."""
    if file_path is None:
        return "No file provided."
    if not os.path.exists(file_path):
        return "File not found."
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in allowed_extensions:
        return f"Unsupported file format: {ext}. Supported: {', '.join(allowed_extensions)}"
    file_size_gb = os.path.getsize(file_path) / (1024 ** 3)
    if file_size_gb > 10:
        return "File too large (>10GB)."
    return None


def process_audio(
    input_file,
    model_name,
    language,
    model_size,
    remove_silence,
    silence_threshold,
    min_silence_duration,
    remove_stutters,
    stutter_patterns,
    output_format,
    words_per_chunk,
    device,
    script_file,
    similarity_threshold,
    burn_subtitles,
    fix_grammar,
    llm_fix,
    llm_url,
    llm_api_key,
    llm_model,
    progress=gr.Progress(),
):
    """Main processing pipeline. Returns (log_text, output_files_list, stats_json, transcription_text)."""
    logs: List[str] = []
    output_files: List[str] = []

    def log(msg: str):
        logs.append(msg)

    try:
        return _run_pipeline(
            input_file, model_name, language, model_size,
            remove_silence, silence_threshold, min_silence_duration,
            remove_stutters, stutter_patterns, output_format,
            words_per_chunk, device, script_file, similarity_threshold,
            burn_subtitles, fix_grammar, llm_fix, llm_url, llm_api_key, llm_model,
            logs, output_files, progress, log,
        )


def _run_pipeline(
    input_file, model_name, language, model_size,
    remove_silence, silence_threshold, min_silence_duration,
    remove_stutters, stutter_patterns, output_format,
    words_per_chunk, device, script_file, similarity_threshold,
    burn_subtitles, fix_grammar, llm_fix, llm_url, llm_api_key, llm_model,
    logs, output_files, progress, log,
):
    """Inner pipeline. Uses a mutable dict to share output_dir with the outer finally block."""
    ctx = {"output_dir": None}

    # --- Validate inputs ---
    if input_file is None:
        return "Error: No input file provided.", [], "{}", ""
    err = validate_file(input_file, SUPPORTED_AUDIO_EXTENSIONS)
    if err:
        return f"Error: {err}", [], "{}", ""

    if script_file is not None and script_file != "":
        err = validate_file(script_file, SUPPORTED_SCRIPT_EXTENSIONS)
        if err:
            log(f"Warning: Invalid script file: {err}. Continuing without verification.")

    progress(0.05, desc="Preparing output directory...")

    base_name = sanitize_filename_stem(Path(input_file).stem)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    ctx["output_dir"] = output_dir
    silence_output = os.path.join(output_dir, "audio_no_silence.wav")

    stats = {
        "input_file": os.path.basename(input_file),
        "model": model_name,
        "language": language,
        "remove_silence": remove_silence,
        "remove_stutters": remove_stutters,
        "silence_stats": None,
        "stutter_stats": None,
        "transcription_stats": None,
        "verification_stats": None,
        "processing_time": 0,
    }

    start_time = time.time()
    working_file = input_file

    # --- Step 1: Silence removal ---
    if remove_silence:
        progress(0.10, desc="Removing silence...")
        log("=== Step 1: Removing Silence ===")
        try:
            success, silence_stats = remove_silence_from_audio(
                input_file, silence_output,
                float(silence_threshold), float(min_silence_duration),
            )
            if success:
                working_file = silence_output
                stats["silence_stats"] = silence_stats
                log(f"Silence removed: {silence_stats['silence_removed']:.1f}s ({silence_stats['silence_ratio']:.1f}%)")
            else:
                log("Warning: Could not remove silence. Continuing with original audio.")
        except Exception as e:
            log(f"Warning: Silence removal failed: {type(e).__name__}")
    else:
        log("=== Step 1: Skipping Silence Removal ===")

    # --- Step 2: Transcribe ---
    progress(0.30, desc="Transcribing audio (this may take a while)...")
    log("=== Step 2: Transcribing Audio ===")
    try:
        words = transcribe_audio(working_file, model_name, model_size, language, device)
    except Exception as e:
        return f"Error during transcription: {type(e).__name__}", [], "{}", ""

    if not words:
        return "Error: No words detected in audio.", [], "{}", ""

    stats["transcription_stats"] = {
        "words_detected": len(words),
        "model_used": model_name,
    }
    log(f"Transcription complete: {len(words)} words detected.")

    # --- Step 3: Stutter removal ---
    if remove_stutters:
        progress(0.65, desc="Removing stutters...")
        log("=== Step 3: Removing Stutters ===")
        patterns = [p.strip() for p in stutter_patterns.split(",") if p.strip()]
        try:
            words, stutter_stats = detect_and_remove_stutters(words, patterns)
            stats["stutter_stats"] = stutter_stats
            log(f"Stutters removed: {stutter_stats['stutters_removed']}")
        except Exception as e:
            log(f"Warning: Stutter removal failed: {type(e).__name__}")
    else:
        log("=== Step 3: Skipping Stutter Removal ===")

    # --- Step 4: Fix grammar/punctuation ---
    if fix_grammar:
        progress(0.72, desc="Fixing grammar & punctuation...")
        log("=== Step 4: Fixing Grammar & Punctuation ===")
        try:
            words = fix_grammar_punctuation(words)
            log("Grammar & punctuation fixed")
        except Exception as e:
            log(f"Warning: Grammar fix failed: {type(e).__name__}")
    elif llm_fix:
        progress(0.72, desc="LLM grammar correction...")
        log("=== Step 4: LLM Grammar Correction ===")
        try:
            words = fix_text_with_llm(words, llm_url, llm_api_key, llm_model)
            log("LLM grammar correction complete")
        except Exception as e:
            log(f"Warning: LLM fix failed: {type(e).__name__}")
    else:
        log("=== Step 4: Skipping Grammar Fix ===")

    # --- Step 5: Generate output ---
    progress(0.80, desc="Generating output files...")
    log("=== Step 5: Generating Output Files ===")

    fmt_lower = output_format.lower()
    try:
        if fmt_lower in ("srt", "all"):
            p = os.path.join(output_dir, f"{base_name}.srt")
            words_to_srt(words, p, int(words_per_chunk))
            output_files.append(p)
        if fmt_lower in ("vtt", "all"):
            p = os.path.join(output_dir, f"{base_name}.vtt")
            words_to_vtt(words, p, int(words_per_chunk))
            output_files.append(p)
        if fmt_lower in ("ass", "all"):
            p = os.path.join(output_dir, f"{base_name}.ass")
            words_to_ass(words, p, int(words_per_chunk))
            output_files.append(p)
        if fmt_lower in ("json", "all"):
            p = os.path.join(output_dir, f"{base_name}.json")
            words_to_json(words, p, int(words_per_chunk))
            output_files.append(p)
    except Exception as e:
        log(f"Warning: Output generation failed: {type(e).__name__}")

    # --- Step 6: Burn subtitles into video ---
    if burn_subtitles:
        progress(0.85, desc="Burning subtitles into video...")
        log("=== Step 5: Burning Subtitles Into Video ===")
        burn_sub_path = None
        if fmt_lower in ("ass", "all"):
            burn_sub_path = os.path.join(output_dir, f"{base_name}.ass")
        elif fmt_lower in ("srt", "all"):
            burn_sub_path = os.path.join(output_dir, f"{base_name}.srt")
        elif fmt_lower in ("vtt", "all"):
            burn_sub_path = os.path.join(output_dir, f"{base_name}.vtt")

        if burn_sub_path:
            video_ext = Path(input_file).suffix
            burn_output = os.path.join(output_dir, f"{base_name}_burned{video_ext}")
            try:
                success = burn_subtitles_to_video(input_file, burn_sub_path, burn_output)
                if success:
                    output_files.append(burn_output)
                    log(f"Video with burned subtitles saved to {os.path.basename(burn_output)}")
                else:
                    log("Warning: Subtitle burning failed")
            except Exception as e:
                log(f"Warning: Subtitle burning failed: {type(e).__name__}")

    # --- Step 7: Verification ---
    if script_file and script_file != "" and os.path.exists(script_file):
        progress(0.90, desc="Verifying against original script...")
        log("=== Step 7: Verifying Against Original Script ===")
        try:
            original_text = parse_script_file(script_file)
            if original_text:
                transcription_text = extract_text_from_words(words)
                is_valid, vstats = verify_transcription(
                    original_text, transcription_text, float(similarity_threshold),
                )
                stats["verification_stats"] = vstats
                log(f"Similarity: {vstats['similarity']:.1f}% (threshold: {vstats['threshold']:.1f}%)")
                log("Verification PASSED" if is_valid else "Verification FAILED - Possible content loss detected")
                if vstats["missing_words"]:
                    log(f"Missing words: {', '.join(vstats['missing_words'][:20])}")
            else:
                log("Warning: Could not parse script file.")
        except Exception as e:
            log(f"Warning: Verification failed: {type(e).__name__}")

    # --- Save stats ---
    end_time = time.time()
    stats["processing_time"] = end_time - start_time

    stats_path = os.path.join(output_dir, f"{base_name}_stats.json")
    try:
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        output_files.append(stats_path)
    except Exception:
        pass

    # --- Summary ---
    log("\n=== Processing Complete ===")
    log(f"Input: {os.path.basename(input_file)}")
    log(f"Model: {model_name}")
    log(f"Language: {language}")
    if stats["silence_stats"]:
        log(f"Silence removed: {stats['silence_stats']['silence_removed']:.1f}s")
    if stats["stutter_stats"]:
        log(f"Stutters removed: {stats['stutter_stats']['stutters_removed']}")
    if stats["verification_stats"]:
        v = stats["verification_stats"]
        log(f"Verification: {'PASSED' if v['is_valid'] else 'FAILED'} ({v['similarity']:.1f}%)")
    log(f"Words: {len(words)}")
    log(f"Processing time: {stats['processing_time']:.1f}s")
    log("\nOutput files:")
    for fp in output_files:
        log(f"  {os.path.basename(fp)}")

    progress(1.0, desc="Done!")
    transcription_text = extract_text_from_words(words)

    return "\n".join(logs), output_files, json.dumps(stats, indent=2, default=str), transcription_text


def build_ui() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(
        title="VideoPolish-er",
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 0.5em; }
        .subtitle { text-align: center; color: #666; margin-top: 0; }
        """,
    ) as app:
        gr.HTML("<h1 class='main-title'>VideoPolish-er</h1>")
        gr.HTML("<p class='subtitle'>Transcription &amp; Audio Cleanup Tool with noise removal and stutter detection</p>")

        with gr.Row():
            # --- Left column: inputs ---
            with gr.Column(scale=1):
                input_file = gr.File(
                    label="Input Audio/Video File",
                    file_types=["audio", "video"],
                    type="filepath",
                )

                with gr.Accordion("Model Settings", open=True):
                    model_name = gr.Dropdown(
                        choices=MODEL_CHOICES,
                        value="faster-whisper",
                        label="Transcription Model",
                    )
                    model_size = gr.Dropdown(
                        choices=MODEL_SIZES,
                        value="large-v3",
                        label="Model Size",
                        allow_custom_value=True,
                    )
                    language = gr.Textbox(
                        value="en",
                        label="Language Code",
                        max_length=10,
                    )
                    device = gr.Dropdown(
                        choices=DEVICE_CHOICES,
                        value="auto",
                        label="Device",
                    )

                with gr.Accordion("Silence Removal", open=False):
                    remove_silence = gr.Checkbox(label="Enable Silence Removal", value=False)
                    silence_threshold = gr.Slider(
                        minimum=-60, maximum=-10, value=-40, step=1,
                        label="Silence Threshold (dB)",
                    )
                    min_silence_duration = gr.Slider(
                        minimum=100, maximum=2000, value=500, step=50,
                        label="Min Silence Duration (ms)",
                    )

                with gr.Accordion("Stutter Removal", open=False):
                    remove_stutters = gr.Checkbox(label="Enable Stutter Removal", value=False)
                    stutter_patterns = gr.Textbox(
                        value="uhm,uh,um,er,ah,hmm,mhm,erm,umm",
                        label="Stutter Patterns (comma-separated)",
                        max_length=500,
                    )

                with gr.Accordion("Output Settings", open=False):
                    output_format = gr.Dropdown(
                        choices=FORMAT_CHOICES,
                        value="srt",
                        label="Output Format",
                    )
                    words_per_chunk = gr.Slider(
                        minimum=1, maximum=8, value=3, step=1,
                        label="Words per Subtitle Chunk",
                    )
                    burn_subtitles = gr.Checkbox(
                        label="Burn Subtitles into Video",
                        value=False,
                        info="Hardcodes subtitles onto the video file using ffmpeg.",
                    )

                with gr.Accordion("Text Fix", open=False):
                    fix_grammar = gr.Checkbox(
                        label="Fix Grammar & Punctuation",
                        value=False,
                        info="Capitalizes 'I', fixes punctuation, capitalizes sentence starts.",
                    )
                    llm_fix = gr.Checkbox(
                        label="Use LLM to Fix Grammar",
                        value=False,
                        info="Sends transcription to an LLM API for grammar correction.",
                    )
                    llm_url = gr.Textbox(
                        value="http://localhost:11434/v1",
                        label="LLM API URL",
                        info="Default is Ollama local. Use OpenAI URL for ChatGPT.",
                    )
                    llm_api_key = gr.Textbox(
                        value="",
                        label="LLM API Key",
                        type="password",
                        info="Leave empty for local Ollama.",
                    )
                    llm_model = gr.Textbox(
                        value="",
                        label="LLM Model Name",
                        info="Leave empty to let the API decide.",
                    )

                with gr.Accordion("Verification", open=False):
                    script_file = gr.File(
                        label="Original Script (optional)",
                        file_types=[".srt", ".vtt", ".ass", ".txt"],
                        type="filepath",
                    )
                    similarity_threshold = gr.Slider(
                        minimum=0, maximum=100, value=80, step=1,
                        label="Similarity Threshold (%)",
                    )

                process_btn = gr.Button("Process", variant="primary", size="lg")

            # --- Right column: outputs ---
            with gr.Column(scale=1):
                transcription_text = gr.Textbox(
                    label="Transcription",
                    lines=12,
                    interactive=False,
                )
                stats_json = gr.Textbox(
                    label="Processing Stats (JSON)",
                    lines=10,
                    interactive=False,
                )
                output_files = gr.File(
                    label="Output Files",
                    file_count="multiple",
                    interactive=False,
                )
                log_output = gr.Textbox(
                    label="Processing Log",
                    lines=14,
                    interactive=False,
                )

        # --- Wire up the button ---
        process_btn.click(
            fn=process_audio,
            inputs=[
                input_file,
                model_name,
                language,
                model_size,
                remove_silence,
                silence_threshold,
                min_silence_duration,
                remove_stutters,
                stutter_patterns,
                output_format,
                words_per_chunk,
                device,
                script_file,
                similarity_threshold,
                burn_subtitles,
                fix_grammar,
                llm_fix,
                llm_url,
                llm_api_key,
                llm_model,
            ],
            outputs=[log_output, output_files, stats_json, transcription_text],
        )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
