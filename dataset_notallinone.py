import numpy as np
import pickle as pkl
import re

import torch
import torch.utils.data as data


def read1to999(n):
    units = [''] + list('십백천')
    nums = '일이삼사오육칠팔구'
    result = []
    i = 0
    while n > 0:
        n, r = divmod(n, 10)
        if r > 0:
            if units[i] == '':
                result.append(nums[r - 1] + units[i])
            else:
                if r == 1:
                    result.append(units[i])
                else:
                    result.append(nums[r - 1] + units[i])
        i += 1
    return ''.join(result[::-1])


def readNumM(n):
    result = ''
    if n >= 1000000000000:
        r, n = divmod(n, 10000000000000)
        tmp = read1to999(r)
        if len(tmp) == 1 and tmp[-1] == '일':
            result += '조'
        else:
            result += tmp + "조"
    if n >= 100000000:
        r, n = divmod(n, 100000000)
        tmp = read1to999(r)
        if len(tmp) == 1 and tmp[-1] == '일':
            result += '억'
        else:
            result += tmp + "억"
    if n >= 10000:
        r, n = divmod(n, 10000)
        tmp = read1to999(r)
        if len(tmp) == 1 and tmp[-1] == '일':
            result += '만'
        else:
            result += tmp + "만"
    result += read1to999(n)
    return result


def readNumK(intNum):
    """
    한글로 숫자 읽기
    """
    korean_num = {"1": "한", "2": "두", "3": "세", "4": "네", "5": "다섯", "6": "여섯", "7": "일곱",
                  '8': "여덟", "9": "아홉", "10": "열", "20": "스물", "30": "서른", "40": "마흔", "50": "쉰",
                  "60": "예순", "70": "일흔", "80": "여든", "90": "아흔"}
    tmp_list = list(korean_num.keys())
    num_list = list()
    for num in tmp_list:
        num_list.append(int(num))
    num_list.sort(reverse=True)
    result = ""
    for num in num_list:
        if intNum >= num:
            intNum -= num
            result += korean_num[str(num)]
    return result


def readLexicon(txt):
    # lexicon 기본 규칙: 참고 사전 <국립국어원 표준국어대사전>
    lexicon_pickle = 'lexicon.pickle'
    with open(lexicon_pickle, 'rb') as handle:
        lexicon = pkl.load(handle)
    sub_pickle = 'sub.pickle'
    # sub 기본 규칙
    with open(sub_pickle, 'rb') as handle:
        sub = pkl.load(handle)
    num_sub_pickle = 'num_sub.pickle'
    with open(num_sub_pickle, 'rb') as handle:
        num_sub = pkl.load(handle)


def removeS(word):
    return word.replace('\\', '').replace('"', '').lower()


def decompose_hangul(text):
    Start_Code, ChoSung, JungSung = 44032, 588, 28
    ChoSung_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    JungSung_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
                     'ㅣ']
    JongSung_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                     'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    line_dec = ""
    line = list(text.strip())

    for keyword in line:
        if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None:
            char_code = ord(keyword) - Start_Code
            char1 = int(char_code / ChoSung)
            line_dec += ChoSung_LIST[char1]
            char2 = int((char_code - (ChoSung * char1)) / JungSung)
            line_dec += JungSung_LIST[char2]
            char3 = int((char_code - (ChoSung * char1) - (JungSung * char2)))
            line_dec += JongSung_LIST[char3]
        else:
            line_dec += keyword
    return line_dec


class Dataset(data.Dataset):
    def __init__(self, sentences, use_lexicon):
        # Load vocabulary
        female_vocab = torch.load(open('/data3/sejikpark/.jupyter/workspace/pyflask_update/models/female_vocab.t7', 'rb'))
        male_vocab = pkl.load(open('/data3/sejikpark/.jupyter/workspace/pyflask_update/models/male_vocab.pkl', 'rb'))
        self.vocab_dict = [female_vocab, male_vocab]
        self.vocab_size = [len(female_vocab), len(male_vocab)]

        # SSML list
        self.ssml = sentences
        self.use_lexicon = use_lexicon
        self.file_name = list(range(len(sentences)))

        self.gen_lu = {'female': 0, 'male': 1}
        self.age_lu = {'age20': 0, 'age30': 1, 'age40': 2}
        self.emo_lu = {'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3, 'surprise': 4, 'fear': 5, 'disgust': 6}

        self.english_word = {'a': '에이', 'b': '비', 'c': '씨', 'd': '디', 'e': '이', 'f': '에프', 'g': '쥐', 'h': '에이치',
                        'i': '아이', 'j': '제이', 'k': '케이', 'l': '엘', 'n': '엔', 'm': '엠', 'o': '오', 'p': '피',
                        'q': '큐', 'r': '얼', 's': '에스', 't': '티', 'u': '유', 'v': '브이', 'w': '더블유', 'x': '엑스',
                        'y': '와이', 'z': '지'}

        num_sub_pickle = '/data3/sejikpark/.jupyter/workspace/pyflask_update/dictionary/num_sub.pickle'
        with open(num_sub_pickle, 'rb') as handle:
            self.num_sub = pkl.load(handle)

        lexicon_pickle = '/data3/sejikpark/.jupyter/workspace/pyflask_update/dictionary/lexicon.pickle'
        with open(lexicon_pickle, 'rb') as handle:
            self.lexicon = pkl.load(handle)

        sub_pickle = '/data3/sejikpark/.jupyter/workspace/pyflask_update/dictionary/sub.pickle'
        with open(sub_pickle, 'rb') as handle:
            self.sub = pkl.load(handle)

    def __len__(self):
        return len(self.ssml)

    def __getitem__(self, idx):
        # default style
        style = self.getstyle()

        # SSML preprocess
        ssml = self.ssml[idx]
        # 문장 앞 뒤 띄어쓰기 지우기
        txt = ssml.text.replace('\\n', '').strip()

        content = ssml.find_all()

        for c in content:
            # 문장별 파라미터: 1. emotion, 2. voice
            if c.name == 'voice':
                if removeS(c.attrs['name']) == '여':
                    style['gender'] = self.gen_lu['female']
                else:
                    style['gender'] = self.gen_lu['male']
            elif c.name == 'emotion':
                style['emotion'] = self.emo_lu[removeS(c.attrs['class'][0])]
            # 문장 처리(현재 return 값): 1. say-as 처리, 2. sub 처리
            elif c.name == 'say-as':
                check = removeS(c.attrs['interpret-as'])
                strNum_list = re.findall('\d+', c.contents[0])
                tmp = c.contents[0]

                for strNum in strNum_list:
                    tmpNum = ""
                    intNum = int(strNum)
                    if check == "한문-분리":
                        for s in strNum:
                            # 한글자씩 읽기 (0 == 공)
                            mandarin_num = {"0": "공", "1": "일", "2": "이", "3": "삼", "4": "사", "5": "오", "6": "육",
                                            "7": "칠", "8": "팔", "9": "구"}
                            tmpNum += mandarin_num[s]
                    elif check == "한문":
                        # 숫자 한문 읽기
                        tmpNum = readNumM(intNum)
                    else:  # check == "한글"
                        # 100이상 한문 읽기 + 이하 한글 읽기
                        tmpNum = readNumM(intNum // 100 * 100) + readNumK(intNum % 100)

                    tmp = c.contents[0].replace(strNum, tmpNum)

                english = re.sub('[^a-zA-Z]', '', c.contents[0])
                if english != '':
                    num_sub = self.num_sub

                    for key, value in num_sub.items():
                        if key in english:
                            tmp.replace(english, value)

                txt = txt.replace(c.contents[0], tmp)

            elif c.name == 'sub':
                tmp = removeS(c.attrs['alias'])
                txt = txt.replace(c.contents[0], tmp)

        # lexicon 처리를 마지막으로
        if self.use_lexicon:
            lexicon = self.lexicon
            sub = self.sub

            word_list = txt.split(' ')
            for k, word in enumerate(word_list):
                english = re.sub('[^a-zA-Z]', '', word)
                if english != '':
                    for key, value in lexicon.items():
                        if key.lower() == english.lower():
                            word_list[k] = word_list[k].replace(english, value)

                    for key, value in sub.items():
                        if key.lower() == english.lower():
                            word_list[k] = word_list[k].replace(english, value)

            txt = word_list[0]
            for word in word_list[1:]:
                txt += ' ' + word

        # 영어 단어 전체 교체

        txt = txt.lower()
        for key, value in self.english_word.items():
            txt = txt.replace(key, value)

        # text vector encoding & ssml about audio
        decompose_list = list()
        for c in content:
            # 음소별 파라미터: 1. prosody, 2. break
            if c.name == 'prosody':
                if 'volume' in c.attrs.keys():  # volume: 1
                    cur = removeS(c.attrs['volume'])
                    cur = cur.lower().replace('%', '')
                    decompose_list.append([decompose_hangul(c.contents[0]), 1, float(cur)])
                elif 'rate' in c.attrs.keys():  # rate: 2
                    cur = removeS(c.attrs['rate'])
                    cur = cur.replace('%', '')
                    decompose_list.append([decompose_hangul(c.contents[0]), 2, float(cur)])
                elif 'pitch' in c.attrs.keys():  # pitch: 3
                    cur = removeS(c.attrs['pitch'])
                    cur = cur.lower().replace('hz', '')
                    decompose_list.append([decompose_hangul(c.contents[0]), 3, float(cur)])
            elif c.name == 'break':
                cur = removeS(c.attrs['time'])
                cur = cur.replace('s', '').replace('/', '')
                decompose_list.append([decompose_hangul(c.contents[0].strip().replace('\\n', '')), 4, float(cur)])

        decompose_txt = decompose_hangul(txt)
        which = [0] * len(decompose_txt)
        how = [0] * len(decompose_txt)

        JungSung_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ',
                         'ㅠ', 'ㅡ', 'ㅢ',
                         'ㅣ']

        for d in decompose_list:
            start = decompose_txt.find(d[0])
            end = start + len(d[0])
            if d[1] == 4:
                which[start] = d[1]
                how[start] = d[2]
            elif d[1] == 2:
                for i in range(start, end):
                    how[i] = d[2]
                    if decompose_txt[i] in JungSung_LIST:
                        which[i] = d[1]
                    else: # 모음이 아닌 경우에 대해 처리
                        which[i] = d[1] * -1
                # which[start:end] = [d[1]] * (end-start)
                # how[start:end] = [d[2]] * (end-start)
            else:
                which[start:end] = [d[1]] * (end-start)
                how[start:end] = [d[2]] * (end-start)

                '''
                if d[1] == 1 or d[1] == 2:
                    step = (d[2]-100) / ((end - start + 1) // 2)
                    nxt_step = 100 + step
                else:
                    step = d[2] / ((end-start+1) // 2)
                    nxt_step = step

                for i in range(start, start + (end-start)//2):
                    how[i] = nxt_step
                    nxt_step += step
                for i in range(start + (end-start)//2-1, end):
                    nxt_step -= step
                    how[i] = nxt_step
                '''

        vocab_dict = self.vocab_dict[style['gender']]
        char2onehot = lambda x: vocab_dict[x] if x in vocab_dict.keys() else None
        decompose_txt = [char2onehot(xx) for xx in decompose_txt]
        for i, d in enumerate(decompose_txt):
            if d is None:
                which[i] = None
                how[i] = None

        txt_feat = [x for x in decompose_txt if x is not None]
        which = [x for x in which if x is not None]
        how = [x for x in how if x is not None]

        attributes = [np.asarray(which), np.asarray(how)]

        file_name = '/data3/sejikpark/.jupyter/workspace/pyflask_update/' + str(self.file_name[idx]) + '.wav'

        return {'origin_txt': txt,
                'txt': np.asarray(txt_feat),
                'attributes': attributes,
                'style': style,
                'filename': file_name}

    def getstyle(self):
        age = self.age_lu['age30']
        gender = self.gen_lu['male']
        emotion = self.emo_lu['neutral']
        return {'age': age, 'gender': gender, 'emotion': emotion}

    def get_vocab_size(self, x):
        return self.vocab_size[x]


if __name__ == '__main__':
    print('dataset')
