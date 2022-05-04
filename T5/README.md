# ke-T5를 사용한 KorQuAD 질문답변 모델  

- 이 코드는 https://analyticsindiamag.com/guide-to-question-answering-system-with-t5-transformer/ 의 코드를 바탕으로 만들어졌습니다. 
- ke-t5의 pretrained model(base) 을 사용하였고, 위키백과 질문에 대해 답변할 수 있도록 후반부 함수를 구현하였습니다. 
- T5 트랜스포머와 Pytorch Lightning을 사용한 모델클래스와 데이터클래스로 구성되어 있습니다. 

## 설치 

- 모델링 

  ```python
  !pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
  !pip install --quiet "torchvision" "torchmetrics>=0.6" "pytorch-lightning>=1.4" "ipython[notebook]" "torch>=1.6, <1.9"
  ```

  

-  예측 (답변생성 함수)

  ```python
  !pip install tensorflow 
  !pip install sklearn 
  !pip install konlpy
  ```

  

## 모델 

### Pretrained models

HuggingFace 모델 허브에 모델이 추가되어 사용이 가능합니다.
Huggingface TFT5ForConditionalGeneration, T5ForConditionalGeneration를 사용하실 경우 아래 코드와 같이 사용하시면 됩니다.

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = 'KETI-AIR/ke-t5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
```



## 데이터셋 

- 본 프로젝트에서는 KorQuAD 1.0, KorQuAD  2.0 데이터를 가공하여 사용했습니다. 
- 한국어 위키백과의 데이터셋인 KorQuAD는 다음 주소에서 다운로드 받을 수 있습니다. (https://korquad.github.io/)
- KorQuAD 2.0의 경우, 내용이 HTML로 되어 있기 때문에 전처리가 필요합니다. 본 프로젝트의 서비스 형태는 챗봇이기 때문에 표 데이터 등이 답변으로 들어가 있는 경우에는 내용 표시가 어려워 표를 제외한 나머지 데이터를 사용했습니다. 

- JSON 형태의 데이터를 DataFrame으로 변환하여 사용했습니다. 함수부에서는 DataFrame 형태의 데이터를 사용합니다. 



## 모델 클래스

- pytorch lightning module 클래스를 생성합니다. 

```python
 class BioDataModule(pl.LightningDataModule):
   def __init__(
       self,
       train_df: pd.DataFrame,
       test_df: pd.DataFrame,
       tokenizer:T5Tokenizer,
       batch_size: int = 8,
       source_max_token_len: int = 396,
       target_max_token_len: int = 32,
       ):
     super().__init__()
     self.train_df = train_df
     self.test_df = test_df
     self.tokenizer = tokenizer
     self.batch_size = batch_size
     self.source_max_token_len = source_max_token_len
     self.target_max_token_len = target_max_token_len
   def setup(self):
     self.train_dataset = BioQADataset(
         self.train_df,
         self.tokenizer,
         self.source_max_token_len,
         self.target_max_token_len
         )
     self.test_dataset = BioQADataset(
     self.test_df,
     self.tokenizer,
     self.source_max_token_len,
     self.target_max_token_len
     )
   def train_dataloader(self):
     return DataLoader(
         self.train_dataset,
         batch_size=self.batch_size,
         shuffle=True,
         num_workers=4
         )
   def val_dataloader(self):
     return DataLoader(
         self.test_dataset,
         batch_size=self.batch_size,
         num_workers=4
         )
   def test_dataloader(self):
     return DataLoader(
         self.test_dataset,
         batch_size=1,
         num_workers=4
         )
 BATCH_SIZE = 4
 N_EPOCHS = 6
 data_module = BioDataModule(train_df, val_df, tokenizer, batch_size=BATCH_SIZE)
 data_module.setup()
```



## T5 모델을 사용하여 Pytorch lightning 모듈 빌드 

```python
 class BioQAModel(pl.LightningModule):
   def __init__(self):
     super().__init__()
     self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
   def forward(self, input_ids, attention_mask, labels=None):
     output = self.model(
         input_ids, 
         attention_mask=attention_mask,
         labels=labels)
     return output.loss, output.logits
   def training_step(self, batch, batch_idx):
     input_ids = batch['input_ids']
     attention_mask=batch['attention_mask']
     labels = batch['labels']
     loss, outputs = self(input_ids, attention_mask, labels)
     self.log("train_loss", loss, prog_bar=True, logger=True)
     return {"loss": loss, "predictions":outputs, "labels": labels}
   def validation_step(self, batch, batch_idx):
     input_ids = batch['input_ids']
     attention_mask=batch['attention_mask']
     labels = batch['labels']
     loss, outputs = self(input_ids, attention_mask, labels)
     self.log("val_loss", loss, prog_bar=True, logger=True)
     return loss
   def test_step(self, batch, batch_idx):
     input_ids = batch['input_ids']
     attention_mask=batch['attention_mask']
     labels = batch['labels']
     loss, outputs = self(input_ids, attention_mask, labels)
     self.log("test_loss", loss, prog_bar=True, logger=True)
     return loss
   def configure_optimizers(self):
     optimizer = AdamW(self.parameters(), lr=0.0001)
     return optimizer
 model = BioQAModel() 
```





## 체크포인트 저장 및 불러오기 

### 체크포인트 생성하고 저장하기 

```python
 checkpoint_callback = ModelCheckpoint(
     dirpath="checkpoints",
     filename="best-checkpoint",
     save_top_k=1,
     verbose=True,
     monitor="val_loss",
     mode="min"
 )
 #logger = TensorBoardLogger("training-logs", name="bio-qa")
 #logger = TensorBoardLogger("training-logs", name="bio-qa")
 trainer = pl.Trainer(
     #logger = logger,
     checkpoint_callback=checkpoint_callback,
     max_epochs=N_EPOCHS,
     gpus=1,
     progress_bar_refresh_rate = 30
 ) 
```

### 저장된 체크포인트 불러오기 

```python
trained_model = BioQAModel.load_from_checkpoint("checkpoints/best-checkpoint.ckpt")
trained_model.freeze() 
```



## 답변생성 함수부분

- in_title(question, dat) 함수에서 인풋 질문에 wiki 문서의 제목이 포함되었는지 체크하고 일치하는 제목의 문서들을 데이터프레임으로 반환합니다. 
- title_re(t_list, question) 함수에서 in_title()에서 반환받은 데이터프레임(인풋 질문과 일치하는 제목의 문서들)과 해당 문서 내 질문리스트와 비교하여 유사도를 측정합니다. 각 문장을 tfidf로 벡터화해 맨하탄거리를 측정했습니다. 측정된 거리가 가장 가까운 내용의 문서들을 반환합니다. 
-  cosin_sim(t_list, question) 함수에서 인풋 질문과 title_re()에서 반환받은 데이터프레임 내의 본문내용을 비교하여 유사도를 측정합니다. 각 문장을 tfidf로 벡터화해 코사인 유사도를 측정했습니다. 코사인 유사도의 값이 가장 높은 내용의 본문 내용을 반환합니다. 
- anw_test(context,question) 함수에서 앞서 학습된 모델을 사용하여 정답을 예측합니다. 예측된 답변과 해당 문서 내 본문 내용을 튜플형태로 반환합니다. 
- last(question, dat) 함수에서 in_title(), title_re()로 타겟 문서의 범위를 좁히고 cosin_sim() 함수에서 인풋 질문과 코사인 유사도가 높은 본문 내용을 context에 저장합니다. anw_test()에서 예측된 답변과 해당 문서 내 본문 내용을 튜플형태로 answer 에 저장해 반환합니다.  







## References

https://analyticsindiamag.com/guide-to-question-answering-system-with-t5-transformer/

https://github.com/AIRC-KETI/ke-t5

