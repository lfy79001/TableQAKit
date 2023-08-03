import json
import os

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")

CAPTION_FILE = 'mmqa_captions.json'
# CAPTION_FILE = 'mmqa_captions_llava.json'
# CAPTION_FILE = 'mmqa_captions_llava_simple.json'

def get_caption_map(file_path=None):
    """
    Get the caption map.
    """
    if not file_path:
        file_path = os.path.join(ROOT_DIR, 'utils', CAPTION_FILE)

    with open(file_path, "r") as f:
        caption_map = json.load(f)
    return caption_map


def get_caption(id):
    """
    Get the caption of the picture by id.
    """
    with open(os.path.join(ROOT_DIR, 'utils', CAPTION_FILE), "r") as f:
        caption = json.load(f)
    if id in caption.keys():
        return caption[id]
    else:
        return ""
