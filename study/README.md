# MeCab설치
### 1. mecab-ko-msvc설치하기 'C 기반으로 만들어진 mecab'이 윈도우에서 실행될 수 있도록 하는 역할
#### 1.1 링크 클릭https://github.com/Pusnow/mecab-ko-msvc/releases/tag/release-0.9.2-msvc-3
#### 1.2 윈도우 버전에 따라 32bit/64bit 선택하여 다운로드
#### 1.3 'C 드라이브'에 mecab폴더 만들기=>"C:/mecab"

### 2. mecab-ko-dic-msvc.zip 기본사전 설치하기
#### 2.1 링크 클릭https://github.com/Pusnow/mecab-ko-dic-msvc/releases/tag/mecab-ko-dic-2.1.1-20180720-msvc
#### 2.2 사전다운로드 'mecab-ko-dic-msvc.zip'
#### 2.3 앞서 '1-3'에서 만들었던 "C:/mecab"에 압축해제

### 3. python wheel설치하기
#### 3.1 링크 클릭 https://github.com/Pusnow/mecab-python-msvc/releases/tag/mecab_python-0.996_ko_0.9.2_msvc-2
#### 3.2 파이썬 및 윈도우 버전에 맞는 whl 다운로드 mecab_python-0.996_ko_0.9.2_msvc-cp37-cp37m-win_amd64.whl
#### 3.3 다운로드 받은 파일을 site-packages폴더에 옮겨놓기
#### 3.2 python 사용자의 경우 cmd창에서 site-package 폴더로 이동하여 
```
pip install mecab_python-0.996_ko_0.9.2_msvc-cp37-cp37m-win_amd64.whl'
```

### 4. mecab 실행해보기
#### 4.1 기본 소스 코드 넣어서 사용하기
```
import MeCab
m = MeCab.Tagger()
out = m.parse("미캅이 잘 설치되었는지 확인중입니다.")
print(out)
```
#### 4.2 결과확인
```
미      NNP,인명,F,미,*,*,*,*
캅      NNP,인명,T,캅,*,*,*,*
이      JKS,*,F,이,*,*,*,*
잘      MAG,*,T,잘,*,*,*,*
설치    NNG,행위,F,설치,*,*,*,*
되      XSV,*,F,되,*,*,*,*
었      EP,*,T,었,*,*,*,*
는지    EC,*,F,는지,*,*,*,*
확인    NNG,행위,T,확인,*,*,*,*
중      NNB,*,T,중,*,*,*,*
입니다  VCP+EF,*,F,입니다,Inflect,VCP,EF,이/VCP/*+ᄇ니다/EF/*
.       SF,*,*,*,*,*,*,*
EOS
```
