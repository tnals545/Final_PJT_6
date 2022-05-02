# KeyBERT

### 목차

- [개요](https://github.com/tnals545/Final_PJT_6/tree/master/KeyBERT#개요)
- [설치](https://github.com/tnals545/Final_PJT_6/tree/master/KeyBERT#설치)
- [모델링](https://github.com/tnals545/Final_PJT_6/tree/master/KeyBERT#모델링)
  - [모델](https://github.com/tnals545/Final_PJT_6/tree/master/KeyBERT#모델)
  - [데이터_로드](https://github.com/tnals545/Final_PJT_6/tree/master/KeyBERT#데이터_로드)
  - [KeyBERT함수_만들기](https://github.com/tnals545/Final_PJT_6/tree/master/KeyBERT#KeyBERT함수_만들기)
    - [find_titles](https://github.com/tnals545/Final_PJT_6/tree/master/KeyBERT#KeyBERT함수_만들기#find_titles)
    - [find_context](https://github.com/tnals545/Final_PJT_6/tree/master/KeyBERT#KeyBERT함수_만들기#find_context)
- [결과확인](https://github.com/tnals545/Final_PJT_6/tree/master/KeyBERT#결과확인)

<br>

## 개요

KeyBERT 모델을 통해 질문에 대한 키워드를 추출하여 질문의 요점을 파악하고, 그에 맞는 title의 context를 반환

<br>

## 설치

SBERT를 위한 패키지인 sentence_transformers와 형태소 분석기 KoNLPy를 설치합니다.

```python
!pip install sentence_transformers
!pip install konlpy
```

<br>

## 모델링

### 모델

embedding에 사용할 SBERT와 토큰화에 필요한 okt를 변수에 저장합니다.

```python
# 필요한 모듈 import
import pandas as pd
import numpy as np
import itertools

from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm.notebook import tqdm

# 모델, okt 저장
model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
okt = Okt()
```

<br>

### 데이터_로드

모델링에 사용할 데이터를 불러옵니다.

```python
# data load
main_df = pd.read_pickle('/content/drive/MyDrive/6조_파이널PJT/data/SBERT/SBERT_final.pkl')
sub_df = pd.read_pickle('/content/drive/MyDrive/6조_파이널PJT/data/SBERT/split_by_doc.pkl')

# stopword load
with open('/content/drive/MyDrive/6조_파이널PJT/data/SBERT/sbert_stop_words.txt', 'r') as file:
    string = file.read()
    stop_words = string.split('\n')
```

<br>

### KeyBERT함수_만들기

- find_titles

질문으로 들어온 문장에서 단어를 추출하여 키워드를 생성하고, 해당 키워드와 관련이 있는 제목을 찾아 최종적으로 소제목과 그 내용을 리스트에 저장하는 함수입니다.

```python
# 질문 -> 제목, 소제목
# output : str, li (제목, 소제목 리스트)

def find_titles(data, stop_words, question):

  # 질문 -> 제목
  tokenized_doc = okt.pos(question) # 형태소 분석기를 통해 명사와 숫자 추출
  tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if ((word[1] == 'Number') | (word[1] == 'Noun')) \
                              & (word[0] not in stop_words)]) # stop_words에 포함 된 문구 제외
  
  # 질문 토큰 embedding
  n_gram_range = (2, 3) # n-gram 추출 값 설정
  try:
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([tokenized_nouns]) 
    qas_candidates = count.get_feature_names_out()
    qas_embedding = model.encode(qas_candidates)
  except:
    qas_candidates = np.array([tokenized_nouns], dtype = object)
    qas_embedding = model.encode(qas_candidates)
    
  # 질문 embedding, 단어들과 코사인 유사도 확인
  qas_doc_embedding = model.encode([question])
  qas_candidate_embeddings = model.encode(qas_candidates)

  top_n = 1 # 코사인 유사도가 높은 상위 n개의 키워드 추출
  distances = cosine_similarity(qas_doc_embedding, qas_candidate_embeddings)
  keywords = [qas_candidates[index] for index in distances.argsort()[0][-top_n:]]
  keyword = keywords[0] # top_n = 1인 경우

  # 키워드 공백 기준으로 분리 (ex. 지미 카터 -> [지미, 카터])
  keys = keyword.split(' ')
  title_li = list(data['title'])
  same = []
  for i in range(len(title_li)):
    for j in range(len(keys)):
      if keys[j] not in title_li[i]:
        break
      elif j == (len(keys)-1): # 타이틀에 키워드가 모두 들어간 경우
        same.append([i,title_li[i]])

  if len(same) == 1: # 키워드가 들어간 타이틀이 1개인 경우
    title = same[0][1] # 해당 문서의 제목 반환

  else: # 제목 리스트에서 코사인 유사도 검사
    key_len = len(keyword.replace(' ',''))
    distances_li = []
    keywords_li = []

    # 키워드가 들어간 타이틀이 2개 이상인 경우
    # 해당 문서들을 title_data로 저장
    if len(same) > 1: 
      title_data = pd.DataFrame(columns = data.columns)
      for i in range(len(same)):
        title_data.loc[i] = data.loc[same[i][0]]

    # 키워드가 들어간 타이틀이 없는 경우
    # 글자 수가 일치하는 문서들을 title_data로 저장
    else: 
      title_data = data.loc[data['title_len']==key_len]
      title_data.reset_index(drop=True, inplace=True)

    # title_data 문서 안에서 코사인 유사도 계산
    title_li = list(title_data['title'])
    for i in tqdm(range(len(title_li))):
      distances = cosine_similarity(qas_embedding, title_data.loc[i,'embedding'])
      distances_li.append(distances)
      keywords_li.append(max(map(max, distances_li[i])))
      title_idx = keywords_li.index(max(keywords_li))
    title = title_li[title_idx] # 유사도 가장 높은 단어를 제목으로 선정

  # 제목 -> 소제목
  dt = data[data['title'] == title]
  subs = dt['sub_name'].values[0]
  subs_idx = dt['sub_idx'].values[0]

  subs_li = ['기본']
  for i in range(len(subs)-1): # 내용 있는 소제목만 출력
    if subs_idx[i+1] - subs_idx[i] != 1:
      subs_li.append(subs[i])
  if len(subs) > 0:
    subs_li.append(subs[len(subs)-1])

  return title, subs_li
```

<br>

- find_context

find_titles 함수에서 return된 title, subs_li 데이터를 통해 사용자의 질문에 알맞는 context를 반환하는 함수입니다.

```python
# 제목, 소제목 -> 내용
# output : str (소제목에 해당하는 내용)

def find_content(data, split_data, title, subtitle):
  dt = data[data['title'] == title]

  # 1) desc -> desc 출력
  if subtitle == '기본':
    return dt['desc'].values[0]

  # 2) 그 외 특정 목차
  # 해당 목차의 내용을 출력
  else:
    try:
      j = dt['sub_name'].values[0].index(subtitle)
      st = dt['sub_idx'].values[0][j] + 1
      try:
        ed = dt['sub_idx'].values[0][j+1]
      except:
        ed = -1
      output = ' '.join(split_data[title].split('\n')[st:ed])
      return output
    except:
      return '잘못된 입력입니다.'
```

<br>

## 결과확인

```python
# 실행
question = input('질문해주세요 : ')

title, sub_li = find_titles(main_df, stop_words, question)
print('title:', title)
print('소제목 리스트\n', ', '.join(sub_li))

sub = input('소제목 선택 : ')
output = find_content(main_df, sub_df, title, sub)
print(output)
```

