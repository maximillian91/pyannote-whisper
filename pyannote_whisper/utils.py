from pyannote.core import Segment, Annotation, Timeline
from whisper.utils import optional_int, optional_float, str2bool, write_txt, write_vtt, write_srt, format_timestamp
from typing import Iterator, TextIO

def get_text_with_timestamp(transcribe_res, duration):
    timestamp_texts = []
    for item in transcribe_res['segments']:
        start = item['start']
        end = min(duration, item['end'])
        text = item['text']
        timestamp_texts.append((Segment(start, end), text))
    return timestamp_texts


def add_speaker_info_to_text(timestamp_texts, ann):
    spk_text = []
    for seg, text in timestamp_texts:
        spk = ann.crop(seg).argmax()
        spk_text.append((seg, spk, text))
    return spk_text


def merge_cache(text_cache):
    sentence = ''.join([item[-1] for item in text_cache])
    spk = text_cache[0][1]
    start = text_cache[0][0].start
    end = text_cache[-1][0].end
    return Segment(start, end), spk, sentence


PUNC_SENT_END = ['.', '?', '!']


def merge_sentence(spk_text):
    merged_spk_text = []
    pre_spk = None
    text_cache = []
    for seg, spk, text in spk_text:
        if spk != pre_spk and pre_spk is not None and len(text_cache) > 0:
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = [(seg, spk, text)]
            pre_spk = spk

        elif text[-1] in PUNC_SENT_END:
            text_cache.append((seg, spk, text))
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = []
            pre_spk = spk
        else:
            text_cache.append((seg, spk, text))
            pre_spk = spk
    if len(text_cache) > 0:
        merged_spk_text.append(merge_cache(text_cache))
    return merged_spk_text


def diarize_text(transcribe_res, diarization_result, duration):
    timestamp_texts = get_text_with_timestamp(transcribe_res, duration)
    spk_text = add_speaker_info_to_text(timestamp_texts, diarization_result)
    res_processed = merge_sentence(spk_text)
    return res_processed


def write_to_txt(spk_sent, file):
    with open(file, 'w') as fp:
        for seg, spk, sentence in spk_sent:
            line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sentence}\n'
            fp.write(line)


def write_txt_with_spk(spk_sent, file: TextIO):
    print("WEBVTT\n", file=file)
    for seg, spk, sentence in spk_sent:
        print(
            f'{seg.start:.2f} {seg.end:.2f} {spk} {sentence}\n',
            file=file,
            flush=True,
        )

def write_vtt_with_spk(spk_sent, file: TextIO):
    print("WEBVTT\n", file=file)
    for seg, spk, sentence in spk_sent:
        print(
            f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n"
            f"<v {spk}>{sentence.strip().replace('-->', '->')}</v>\n",
            file=file,
            flush=True,
        )

# def write_vtt_with_spk(transcript: Iterator[dict], file: TextIO):
#     print("WEBVTT\n", file=file)
#     for segment in transcript:
#         print(
#           f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
#           f"<v {segment['speaker']}>{segment['text'].strip().replace('-->', '->')}</v>\n",
#           file=file,
#           flush=True,
#         )


# def write_to_vtt_with_spk(spk_sent, file):
#     with open(file, 'w') as fp:
#         for seg, spk, sentence in spk_sent:
#             line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sentence}\n'
#             fp.write(line)



