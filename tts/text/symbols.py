""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from text import cmudict

_punctuation = '!\'",.:;? '
_math = '#%&*+-/[]()'
_special = '_@©°½—₩€$'
_accented = 'áçéêëñöøćž'
_numbers = '0123456789'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as
# uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

_korean = ['ㅓ', 'ㅝ', 'ㅃ', 'ㅛ', 'ㅢ', 'ㄶ', 'ㅇ', 'ㅎ', 'ㅖ', 'ㅗ', 'ㅠ', 'ㅆ', 'ㅜ', 'ㅌ', 'ㄿ', 'ㅔ', 'ㅋ', 'ㄲ', 'ㅑ', 'ㄸ','ㅙ', 'ㅞ', 'ㅅ', 'ㅘ', 'ㄻ', 'ㅍ', 'ㄳ', 'ㄼ', 'ㄹ', 'ㅄ', 'ㅡ', 'ㅈ', 'ㅂ', 'ㅣ', 'ㅟ', 'ㄽ', 'ㅐ', 'ㅀ', 'ㅕ', 'ㅒ', 'ㄷ', 'ㅏ', 'ㅊ', 'ㄺ', 'ㄴ', 'ㄱ', 'ㅉ', 'ㄵ', 'ㅁ', 'ㄾ', 'ㅚ']

# Export all symbols:
symbols = list(_punctuation) + _korean + list(_math + _special + _accented + _numbers + _letters) + _arpabet
