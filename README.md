# LSTM



## 순환신경망 (Recurrent Neural Network, RNN)

- RNN은 입력과 출력을 시퀀스 단위로 처리하는 모델.
- 번역기를 생각해보면, 입력에 사용되는 데이터인 문장도 결국 단어의 시퀀스이고 출력도 단어의 시퀀스.
- 이러한 시퀀스를 적절하게 처리하기 위해 고안된 모델들을 시퀀스 모델이라고 함.
- RNN은 그 중 기초가 되는 모델이라고 할 수 있고, 기존에 알던 신경망처럼 단순히 은닉층에서 출력층 방향으로만 값이 전달되는 것이 아니라, '메모리 셀'이라는 특수한 은닉층이 존재하여 이전의 값을 기억하는 일종의 메모리 역할을 수행.
- 이러한 메모리 셀은 다음시점의 자신에게 'hidden state'라고 불리는 특수한 값을 전달해 활용할 수 있도록 함.
- 위키독스 : 딥 러닝을 이용한 자연어 입문 처리 (https://wikidocs.net/22886)



## LSTM (Long Short-Term Memory, LSTM)

![](LSTM.assets/lstm.png)

- 기초 RNN의 한계 : RNN의 시점이 길어지면 길어질수록 기존에 가지고 있던 정보가 뒤로 충분히 전달되지 못하는 형상이 발생.
-  중요한 정보가 앞쪽에 존재할 수도 있는데 그러한 정보가 충분히 전달되지 못한다는 것은 성능에 치명적인 영향을 줌
- 따라서 이러한 '장기 의존성 문제'를 극복하기 위해 'hidden state'와 더불어 장기기억을 위한 정보인 'cell state'를 추가적으로 가짐. 이것이 LSTM의 핵심.
- 시그모이드 함수와 하이퍼볼릭탄젠트 함수 등을 이용해 이러한 값들을 적절히 조절해가면서 정보를 다음 시점으로 전달.
- 위키독스 : 딥 러닝을 이용한 자연어 입문 처리 (https://wikidocs.net/22888)



### Data

```python
import numpy as np
import re
import pandas as pd
import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score

chatbot = pd.read_csv("/content/drive/MyDrive/6조_파이널PJT/data/wiki_qna.csv")

chatbot['q'] = chatbot['q'].str.replace("[^\w]", " ")
chatbot['a'] = chatbot['a'].str.replace("[^\w]", " ")
# 데이터를 불러오고 Q, A 데이터 모두에서 숫자 혹은 문자가 아닌 문자들을 제거한다.
```



### 토큰화

```python
# input과 output 각각 tokenizer 객체를 생성해서 fit시키고 토큰화(인덱싱)한다.

tokenizer_q = Tokenizer()
tokenizer_q.fit_on_texts(encoder_input)
encoder_input = tokenizer_q.texts_to_sequences(encoder_input)

tokenizer_a = Tokenizer()
tokenizer_a.fit_on_texts(decoder_input)
tokenizer_a.fit_on_texts(decoder_output)
decoder_input = tokenizer_a.texts_to_sequences(decoder_input)
decoder_output = tokenizer_a.texts_to_sequences(decoder_output)

# 각 문장의 길이를 맞추기 위해 패딩을 추가한다. mex_len을 따로 명시하지 않으면 자동으로 인풋값 중 최대길이에 맞춰진다.
# padding = "post" 옵션은 0이 뒤쪽에 붙도록 해준다. (0이 앞에 붙으면 필요없는 정보를 먼저 확인하게 되므로)

encoder_input = pad_sequences(encoder_input, padding="post")
decoder_input = pad_sequences(decoder_input, padding="post")
decoder_output = pad_sequences(decoder_output, padding="post")

# 아웃풋(대답)의 단어들에 대한 인덱싱을 불러온다.
# a_to_index는 단어를 인덱스화하고, index_to_a는 인덱스를 단어화한다.

a_to_index = tokenizer_a.word_index
index_to_a = tokenizer_a.index_word

# 데이터의 크기를 보고 적당한 크기로 나눠 학습 데이터셋과 테스트 데이터셋은 나눠 제작한다.

test_size = 2500
encoder_input_train = encoder_input[:-test_size]
decoder_input_train = decoder_input[:-test_size]
decoder_output_train = decoder_output[:-test_size]

encoder_input_test = encoder_input[-test_size:]
decoder_input_test = decoder_input[-test_size:]
decoder_output_test = decoder_output[-test_size:]
```



### 모델링

```python
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Masking
from tensorflow.keras.models import Model

# 인코더 신경망 설계
# Input : 15(패딩 포함 질문 문장길이)를 입력으로 받는다.
# Embedding: len(tokenizer_q.word_index)+1개(패딩값포함)의 인덱스로 되어있는 정보를 50차원으로 임베딩한다.
# Masking : 패딩값에 해당하는 0 정보를 거르기 위해 사용된다. mask_value에 해당하는 값을 제거한다.
# LSTM : 단어를 순환신경망에 넣어 encoder_outputs, h_state, c_state을 리턴하도록 한다.

encoder_inputs = Input(shape=(117,))
encoder_embed = Embedding(len(tokenizer_q.word_index)+1, 50)(encoder_inputs)
encoder_mask = Masking(mask_value=0)(encoder_embed)
encoder_outputs, h_state, c_state = LSTM(50, return_state=True)(encoder_mask)

# 디코더 신경망 설계
# Input : 22(패딩 포함 대답 문장길이)를 입력으로 받는다.
# Embedding: len(tokenizer_a.word_index)+1개(패딩값포함)의 인덱스로 되어있는 정보를 50차원으로 임베딩한다.
# Masking : 패딩값에 해당하는 0 정보를 거르기 위해 사용된다. mask_value에 해당하는 값을 제거한다.
# LSTM : 단어를 순환신경망에 넣어 decoder_outputs를 리턴하도록 한다. 초기 상태값으로 주어진 h_state, c_state를 활용한다.
# Dense : 단어별 인덱스 확률을 뽑아낸다. (softmax 사용)

decoder_inputs = Input(shape=(128,))
decoder_embed = Embedding(len(tokenizer_a.word_index)+1, 50)(decoder_inputs)
decoder_mask = Masking(mask_value=0)(decoder_embed)

decoder_lstm = LSTM(50, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_mask, initial_state=[h_state, c_state])

decoder_dense = Dense(len(tokenizer_a.word_index)+1, activation='softmax')
decoder_softmax_outputs = decoder_dense(decoder_outputs)

# 함수형 케라스를 통해 최종적으로 모델을 제작한다. inputs로 [encoder_inputs, decoder_inputs], outputs로 decoder_softmax_outputs를 준다.
# 이렇게 생성된 모델을 컴파일 및 학습 데이터에 대해 학습시켜 완성한다.
# 이렇게 완성된 신경망들은 '학습'에 사용된다. 각 레이어가 이러한 과정을 통해 학습되기 때문에 추후 실질적 예측에서는 이 신경망의 일부를 가져와 활용한다.

model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])
model.fit(x = [encoder_input_train, decoder_input_train], y = decoder_output_train, validation_data = ([encoder_input_test, decoder_input_test], decoder_output_test), batch_size = 128, epochs = 100)
```



### 결과 확인

```python
# 인코딩 결과로 발생할 상태값도 가져오기 위해 그를 반환할 모델 (encoder_model)

encoder_model = Model(encoder_inputs, [h_state, c_state])

# Input : 디코더 모델을 만들건데, 디코더 모델에 초기값으로 넣을 상태값의 모양을 지정한다. (앞서 확인한 결과 상태값은 (50,)의 형태도 지정되어 있다. 따라서 shape=(50,))
# decoder_lstm : 그러한 상태값들을 초기값으로 쓰고, 앞서 지정한 decoder_mask함수 케라스를 활용해 새로운 결과값과 상태치를 가져온다.
# decoder_dense : 결과치를 기반으로 소프트맥스 결과를 뽑아내 단어를 찾아낼 수 있도록 한다.
# 이것을 모델화하여 사용한다 (decoder_model)

encoder_h_state = Input(shape=(50,))
encoder_c_state = Input(shape=(50,))

pd_decoder_outputs, pd_h_state, pd_c_state = decoder_lstm(decoder_mask, initial_state=[encoder_h_state, encoder_c_state])

pd_decoder_softmax_outputs = decoder_dense(pd_decoder_outputs)

decoder_model = Model([decoder_inputs, encoder_h_state, encoder_c_state], [pd_decoder_softmax_outputs, pd_h_state, pd_c_state])

# 최종 예측을 수행한다.
# 먼저 encoder_model로 input의 최종 상태값을 얻어낸다.
# 그리고 <start>에 해당하는 인덱스를 (1,1)의 numpy 배열에 할당하고 decoding 수행을 시작한다. 초기 상태값은 인코딩 결과로 받은 상태값이다. 이러한 결과로 예측 단어와 새로운 상태값이 도출될 것이다. 또 다시 그를 기반으로 decoding을 수행한다. 이를 반복하다가 <end>가 예측 단어로 확인되면 반복을 멈춘다.
# 결과는 계속 decoded_stc에 추가해준다. 마지막엔 join을 통해 한번에 결과문을 출력해준다.

input_stc = input()
token_stc = input_stc.split()
encode_stc = tokenizer_q.texts_to_sequences([token_stc])
pad_stc = pad_sequences(encode_stc, maxlen=117, padding="post")

states_value = encoder_model.predict(pad_stc)

predicted_seq = np.zeros((1,1))
predicted_seq[0, 0] = a_to_index['<start>']

decoded_stc = []

while True:
    output_words, h, c = decoder_model.predict([predicted_seq] + states_value)

    predicted_word = index_to_a[np.argmax(output_words[0,0])]

    if predicted_word == '<end>':
        break

    decoded_stc.append(predicted_word)

    predicted_seq = np.zeros((1,1))
    predicted_seq[0,0] = np.argmax(output_words[0,0])

    states_value = [h, c]

print(' '.join(decoded_stc))
```



