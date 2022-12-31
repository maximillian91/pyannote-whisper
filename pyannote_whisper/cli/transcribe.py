import argparse
import os
import warnings
import time

import wave
import contextlib
import webvtt
from datetime import datetime
import time
import re
import subprocess

import numpy as np
import torch

from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from whisper.utils import optional_int, optional_float, str2bool, write_txt, write_vtt, write_srt
from whisper.transcribe import transcribe

from pyannote_whisper.utils import diarize_text, write_to_txt, write_vtt_with_spk


timestamp_regex = r'([0-9]{2})\:([0-9]{2})\:([0-9]{2})\.([0-9]{3})'
map_to_sec = np.array([3600, 60, 1, 0.001])
timestamp_to_sec = lambda x: np.dot(map_to_sec, np.array(list(re.search(timestamp_regex, x).groups())).astype(int))

def cli():
    from whisper import available_models

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", nargs="+", type=str, help="audio file(s) to transcribe")
    parser.add_argument("--model", default="small", choices=available_models(), help="name of the Whisper model to use")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="the path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device to use for PyTorch inference")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--verbose", type=str2bool, default=True,
                        help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"],
                        help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None,
                        choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
                        help="language spoken in the audio, specify None to perform language detection")

    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5,
                        help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=5,
                        help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None,
                        help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None,
                        help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1",
                        help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None,
                        help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=True,
                        help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    parser.add_argument("--fp16", type=str2bool, default=True,
                        help="whether to perform inference in fp16; True by default")

    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2,
                        help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4,
                        help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0,
                        help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6,
                        help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    parser.add_argument("--threads", type=optional_int, default=0,
                        help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")
    parser.add_argument("--diarization", type=str2bool, default=True,
                        help="whether to perform speaker diarization; True by default")
    parser.add_argument("--num_speakers", type=optional_int, default=None,
                        help="number of speakers used by pyannote diarization; default is None, where the number will be determined automatically.")
    parser.add_argument("--reuse_transcript", type=str2bool, default=False,
                        help="whether to reuse transcription files in --transcript_dir; False by default")
    parser.add_argument("--transcript_dir", "-v", type=str, default=".", help="directory to find .vtt files")

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    model_dir: str = args.pop("model_dir")
    output_dir: str = args.pop("output_dir")
    device: str = args.pop("device")
    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en") and args["language"] not in {"en", "English"}:
        if args["language"] is not None:
            warnings.warn(
                f"{model_name} is an English-only model but receipted '{args['language']}'; using English instead.")
        args["language"] = "en"

    temperature = args.pop("temperature")
    temperature_increment_on_fallback = args.pop("temperature_increment_on_fallback")
    if temperature_increment_on_fallback is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
    else:
        temperature = [temperature]

    threads = args.pop("threads")
    if threads > 0:
        torch.set_num_threads(threads)

    from whisper import load_model
    model = load_model(model_name, device=device, download_root=model_dir)

    diarization = args.pop("diarization")
    if diarization:
        from pyannote.audio import Pipeline
        # My HuggingFace Token
        use_auth_token = "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"
        # # Original authors HuggingFace Token
        # use_auth_token = "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"

        num_speakers = args.pop("num_speakers")

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=use_auth_token
        )

    reuse_transcript = args.pop("reuse_transcript")
    transcript_dir = args.pop("transcript_dir")

    for audio_path in args.pop("audio"):
        audio_basename = os.path.basename(audio_path)
        audio_dirname = os.path.dirname(audio_path)

        if audio_path[-3:] != 'wav':
            wav_filepath = os.path.join(audio_path + ".wav")
            subprocess.call(['ffmpeg', '-i', audio_path, wav_filepath, '-y'])
        else:
            wav_filepath = audio_path

        with contextlib.closing(wave.open(wav_filepath,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        
        vtt_filepath = os.path.join(transcript_dir, audio_basename + ".vtt")

        if not (os.path.exists(vtt_filepath) and reuse_transcript):
            t0 = time.time()
            result = transcribe(model, wav_filepath, temperature=temperature, **args)
            t1 = time.time()
            print("{:.4f}s".format(t1-t0), "spent on", "result = transcribe(model, wav_filepath, temperature=temperature, **args)")

            # save TXT
            with open(os.path.join(output_dir, audio_basename + ".txt"), "w", encoding="utf-8") as txt:
                write_txt(result["segments"], file=txt)

            # save VTT
            with open(os.path.join(output_dir, audio_basename + ".vtt"), "w", encoding="utf-8") as vtt:
                write_vtt(result["segments"], file=vtt)

            # save SRT
            with open(os.path.join(output_dir, audio_basename + ".srt"), "w", encoding="utf-8") as srt:
                write_srt(result["segments"], file=srt)

            transcribe_segments = result["segments"]
        else:

            transcribe_segment_list = webvtt.read(vtt_filepath)
            
            transcribe_segments = [
                {
                    "start": timestamp_to_sec(item.start),
                    "end": timestamp_to_sec(item.end),
                    "text": item.text
                }
                for item in transcribe_segment_list
            ]

        if diarization:

            t0 = time.time()
            diarization_result = pipeline(wav_filepath, num_speakers=num_speakers)
            t1 = time.time()
            print("{:.4f}s".format(t1-t0), "spent on", "diarization_result = pipeline(wav_filepath)")
            
            spk_txt_filepath = os.path.join(output_dir, audio_basename + "_spk.txt")
            spk_vtt_filepath = os.path.join(output_dir, audio_basename + "_spk.vtt")

            t0 = time.time()
            res = diarize_text(transcribe_segments, diarization_result, duration)
            t1 = time.time()
            print("{:.4f}s".format(t1-t0), "spent on", "res = diarize_text(transcribe_segments, diarization_result)")

            t0 = time.time()
            write_to_txt(res, spk_txt_filepath)
            t1 = time.time()
            print("{:.4f}s".format(t1-t0), "spent on", "write_to_txt(res, txt_filepath)")

            t0 = time.time()
            with open(spk_vtt_filepath, "w", encoding="utf-8") as vtt:
                write_vtt_with_spk(res, file=vtt)
            t1 = time.time()
            print("{:.4f}s".format(t1-t0), "spent on", "write_vtt_with_spk(res, file=vtt)")


if __name__ == '__main__':
    cli()
