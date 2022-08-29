import re
from unidecode import unidecode
import pyopenjtalk
from janome.tokenizer import Tokenizer

# Regular expression matching Japanese without punctuation marks:
_japanese_characters = re.compile(
    r'[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]')

# Regular expression matching non-Japanese characters or punctuation marks:
_japanese_marks = re.compile(
    r'[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]')

# Tokenizer for Japanese
tokenizer = Tokenizer()


def japanese_tokenization_cleaners(text):
    '''Pipeline for tokenizing Japanese text.'''
    words = []
    for token in tokenizer.tokenize(text):
        if token.phonetic != '*':
            words.append(token.phonetic)
        else:
            words.append(token.surface)
    text = ''
    for word in words:
        if re.match(_japanese_characters, word):
            if word[0] == '\u30fc':
                continue
            if len(text) > 0:
                text += ' '
            text += pyopenjtalk.g2p(word, kana=False).replace(' ', '')
        else:
            text += unidecode(word).replace(' ', '')
    if re.match('[A-Za-z]', text[-1]):
        text += '.'
    return text


def japanese_to_romaji_with_accent(text):
    '''Reference https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html'''
    sentences = re.split(_japanese_marks, text)
    marks = re.findall(_japanese_marks, text)
    text = ''
    for i, sentence in enumerate(sentences):
        if re.match(_japanese_characters, sentence):
            if text != '':
                text += ' '
            labels = pyopenjtalk.extract_fullcontext(sentence)
            for n, label in enumerate(labels):
                phoneme = re.search(r'\-([^\+]*)\+', label).group(1)
                if phoneme not in ['sil', 'pau']:
                    text += phoneme.replace('ch', 'ʧ').replace('sh', 'ʃ').replace('cl', 'Q')
                else:
                    continue
                n_moras = int(re.search(r'/F:(\d+)_', label).group(1))
                a1 = int(re.search(r"/A:(\-?[0-9]+)\+", label).group(1))
                a2 = int(re.search(r"\+(\d+)\+", label).group(1))
                a3 = int(re.search(r"\+(\d+)/", label).group(1))
                if re.search(r'\-([^\+]*)\+', labels[n + 1]).group(1) in ['sil', 'pau']:
                    a2_next = -1
                else:
                    a2_next = int(re.search(r"\+(\d+)\+", labels[n + 1]).group(1))
                # Accent phrase boundary
                if a3 == 1 and a2_next == 1:
                    text += ' '
                # Falling
                elif a1 == 0 and a2_next == a2 + 1 and a2 != n_moras:
                    text += '↓'
                # Rising
                elif a2 == 1 and a2_next == 2:
                    text += '↑'
        if i < len(marks):
            text += unidecode(marks[i]).replace(' ', '')
    return text


def japanese_cleaners(text):
    text = japanese_to_romaji_with_accent(text)
    if re.match('[A-Za-z]', text[-1]):
        text += '.'
    return text


def japanese_cleaners2(text):
    return japanese_cleaners(text).replace('ts', 'ʦ').replace('...', '…')
