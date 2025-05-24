## 멀티 모달 파인튜닝

### 프로젝트
~~~
llava 논문을 참고해 멀티 모달 모델을 구현하는 것이 목표
데이터셋은 llava 데이터셋 사용
~~~
LLaVA paper [https://arxiv.org/pdf/2304.08485]  
LLaVA-CC3M-Pretrain-595K[https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K]

### 다이어그램
![[Pasted image 20240831173924.png]](./image/diagram.jpg)  


- 기본구조 
	1. 이미지 > vision 모델 > projection 레이어> 이미지 임베딩 생성
	2. 텍스트 > langauge 모델 임베딩 > 텍스트 임베딩 생성
	3. 이미지 임베딩, 텍스트 임베딩 > concat 임베딩
	4. concat 임베딩 > langauge 모델 어텐션 레이어 입력 > 문장 생성
	5. 학습 과정을 통해 projection 레이어 weight 학습

- 모델 : vision-sigLIP, langauge-llama3

- siglip의 아웃 임베딩사이즈와 llama의 인풋 임베딩 사이즈가 달라 연결할 수 있는 projection layer 생성

```python
class ProjectionModule(nn.Module):
    def __init__(self, mm_hidden_size, hidden_size):
        super(ProjectionModule, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(mm_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
    def forward(self, x):
        return self.model(x)
```

### 학습 데이터셋 생성
- llava 데이터셋은 이미지와 이미지에 대응하는 질의응답 텍스트들로 구성되어 있다.  
- 학습을 위해 아래 템플릿과 같이 수정하고, 각 하나의의 질의 마다 이미지 토큰을 추가해 프롬프트 텍스트를 생성  
  
```python
LLAVA_CHAT_TEMPLATE = """{% for message in messages %} \
  {% if message['from'] == 'human' %}
    USER: {{ message['value'] }} \
  {% else %}
    ASSISTANT: {{ message['value'] }} \
  {% endif %} \
  {% if message['from'] == 'gpt' %} \
  {% else %} \
      {{ eos_token }} \
  {% endif %} \
{% endfor %}"""
```

- 프롬프트 텍스트를 토큰화 후 임베딩모델에 넣어 임베딩을 생성하는데, 이미지 토큰위치는 기준으로 분할 후 이미지 임베딩을 넣어 concat하여 임베딩을 연결
- 연결할 임베딩을 attention 레이어에 인풋으로 입력
- 최종 모델 output은 shift한 라벨과 cross entropy loss로 학습 진행한다.




### next step
- 프리트레인 데이터로 projection layer를 먼저 파인튜닝 후 테이블 QA 데이터로 llm 재학습이 필요함 
- 일정 이상 성능을 위해 많은 학습 시간 및 짜임새있는 학습 전략 필요함, 
- 비용 이슈로 학습 중단 > 추후 학습 재개