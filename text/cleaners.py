""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import re
from unidecode import unidecode
# from phonemizer import phonemize
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)
from indic_transliteration import sanscript

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]


# List of (iast, ipa) pairs:
_iast_to_ipa = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('a', 'ə'),
    ('ā', 'aː'),
    ('ī', 'iː'),
    ('ū', 'uː'),
    ('ṛ', 'ɹ`'),
    ('ṝ', 'ɹ`ː'),
    ('ḷ', 'l`'),
    ('ḹ', 'l`ː'),
    ('e', 'eː'),
    ('o', 'oː'),
    ('k', 'k⁼'),
    ('k⁼h', 'kʰ'),
    ('g', 'g⁼'),
    ('g⁼h', 'gʰ'),
    ('ṅ', 'ŋ'),
    ('c', 'ʧ⁼'),
    ('ʧ⁼h', 'ʧʰ'),
    ('j', 'ʥ⁼'),
    ('ʥ⁼h', 'ʥʰ'),
    ('ñ', 'n^'),
    ('ṭ', 't`⁼'),
    ('t`⁼h', 't`ʰ'),
    ('ḍ', 'd`⁼'),
    ('d`⁼h', 'd`ʰ'),
    ('ṇ', 'n`'),
    ('t', 't⁼'),
    ('t⁼h', 'tʰ'),
    ('d', 'd⁼'),
    ('d⁼h', 'dʰ'),
    ('p', 'p⁼'),
    ('p⁼h', 'pʰ'),
    ('b', 'b⁼'),
    ('b⁼h', 'bʰ'),
    ('y', 'j'),
    ('ś', 'ʃ'),
    ('ṣ', 's`'),
    ('r', 'ɾ'),
    ('l̤', 'l`'),
    ('h', 'ɦ'),
    ("'", ''),
    ('~', '^'),
    ('ṃ', '^')
]]


def devanagari_to_ipa(text):
    text = text.replace('ॐ', 'ओम्')
    text = re.sub(r'\s*।\s*$', '.', text)
    text = re.sub(r'\s*।\s*', ', ', text)
    text = re.sub(r'\s*॥', '.', text)
    text = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
    for regex, replacement in _iast_to_ipa:
        text = re.sub(regex, replacement, text)
    text = re.sub('(.)[`ː]*ḥ', lambda x: x.group(0)
                  [:-1]+'h'+x.group(1)+'*', text)
    return text
    
    
def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def expand_numbers(text):
  return normalize_numbers(text)


def lowercase(text):
  return text.lower()


def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
  return unidecode(text)


def basic_cleaners(text):
  '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def transliteration_cleaners(text):
  '''Pipeline for non-English text that transliterates to ASCII.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def english_cleaners(text):
  '''Pipeline for English text, including abbreviation expansion.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  phonemes = phonemize(text, language='en-us', backend='espeak', strip=True)
  phonemes = collapse_whitespace(phonemes)
  return phonemes


def english_cleaners2(text):
  '''Pipeline for English text, including abbreviation expansion. + punctuation + stress'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  # phonemes = phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)
  phonemes = global_phonemizer.phonemize([text], strip=True, njobs=1)[0]
  phonemes = collapse_whitespace(phonemes)
  return phonemes

def hindi_cleaners2(text):
  '''Pipeline for English text, including abbreviation expansion. + punctuation + stress'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  # phonemes = phonemize(text, language='hi', backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)
  phonemes = global_phonemizer.phonemize([text], strip=True, njobs=1)[0]
  phonemes = collapse_whitespace(phonemes)
  return phonemes

def he_cleaners(text):
    hindi_texts = re.findall(r'\[HI\].*?\[HI\]', text)
    english_texts = re.findall(r'\[EN\].*?\[EN\]', text)
    for hindi_text in hindi_texts:
        cleaned_text = devanagari_to_ipa(hindi_text[4:-4])
        text = text.replace(hindi_text, cleaned_text+' ', 1)
    for english_text in english_texts:
        cleaned_text = english_cleaners2(english_text[4:-4])
        text = text.replace(english_text, cleaned_text+' ', 1)
    text = text[:-1]
    if re.match(r'[^\.,!\?\-…~]', text[-1]):
        text += '.'
    return text
