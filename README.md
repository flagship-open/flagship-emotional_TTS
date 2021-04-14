# Flagship_5Y_M2_1-Emotional_TTS

버전: 0.95
작성자: 박세직
히스토리:
2020/07/03, 초안작성
2020/10/29, 모델 업데이트

***

#### Note

* (2020/07/03) 7월 마스터 버전이 업데이트 되었습니다.
* (2020/10/29) vocoder 변경 및 monotonic attention 활용 추가
* (2020/10/29) 일부 음성에 대한 GST 세부 컨트롤 추가

***

#### System/SW Overview

* 개발목표: 감정이 들어간 자연스러운 발화가 가능한 음성합성모델
* 최종 결과물: Tacotron2 base mel spectrogram 합성 후 VocGAN을 통한 음성합성

***

#### How to Install

* pip install -r requirements.txt -r requirements-local.txt

***

#### (필수) Main requirement

* OS: Ubuntu 18.04 or 16.04
* Container : Docker community 19.03, Nvidia-docker v2.*
* GPU driver : Nvidia CUDA 10.2
* 파이썬 : Python3.7.5
* 프레임워크: Pytorch 1.5
* 이 외 requirements.txt 참조

***

#### (필수) Network Architecture and features

* **Model**
1. Tacotron2를 통한 txt2spectrogram 합성
2. VocGAN을 통한 spectrogram2wav 합성
* **Tacotron2 구조**
1. CNN과 biLSTM으로 이루어진 Encoder
2. Location Sensitive Attention
3. 2개의 LSTM으로 이루어진 Decoder 
4. 감정, 화자, 언어를 control 할 수 있는 embedding
5. prosody를 control 할 수 있는 GST
* **VocGAN 구조**

***

#### (필수) Quick start

* Step1. GPU Version - 호스트 머신에서 아래 명령을 실행한다. 
```
export CUDA_VISIBLE_DEVICES=<GPUID>
python flask_server.py [--port <PORT>]
```
* Step2. GPU Version - 다음과 같은 response를 얻는다.
```
{'response': 음성파일}
```

***

#### (선택) Training Data

***

#### (선택) Training Model

***

#### (선택) Validation metrics calculation

***

#### (필수) HTTP-server API description

* **path, parameter, response를 명시한다.**

*  /
* JSON parameters are:

|Parameter|Type|Description|
|---|---|---|
|sentence|string|message for generating speech|
|emotion|int|One of 10001~10007. An emotion to condition the response on. Optional param, if not specified, 10005 is used|
|gender|int, one of enum|One of 30001~30002. A speaker to condition the response on. Optional param, if not specified, 30001 is used|

* Request
```
POST /
data: {
'sentence': '안녕, 나와 대화하는 법을 알려줄게',
"emotion": '10005',
"age": '20003',
"gender": '30001',
"intensity": 1
}
```

* Response OK
```
200 OK
{
'zip': (음성, attention 관련 데이터)
}
```

***

#### (필수) Repository overview

* `dictionary/` – character를 phoneme로 바꾸는 dictionary
* `models/` – 사용하는 model parameter 저장 공간
* `tts/` – tacotron2 모델 관련
* `VocGAN/` – VocGAN 모델 관련

#### (선택) configuration settings

***