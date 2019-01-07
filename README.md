# 상품 카테고리 분류기

이 문서는 1월 14일 이전까지 계속 수정 될 것입니다.
카카오 아레나에 참여한 라임로봇 닉네임의 소스코드입니다. 
본 분류기는 상품의 타이틀(product)과 이미지 특징(img_feat)만 입력으로 활용하여 대/중/소/상세 카테고리를 예측합니다. 


## 의존성
모든 의존 패키지는 아래의 명령어로 설치할 수 있습니다.
```bash
pip install -r requirements.txt
```
참고로 pytorch 1.0 버전에서만 동작을 확인하였습니다. 즉, 이 외의 버전에선 동작을 보장할 수 없습니다.


## 실행 방법

### 단계 1: 데이터 다운로드
카카오 아레나 - 쇼핑몰 상품 카테고리 분류 대회의 데이터를 다운로드 받습니다.
소스코드 디렉터리(product-categories-classification)의 상위 디렉터리에 dataset으로 저장합니다.

### 단계 2: 데이터 준비 및 보카 생성
모델의 학습 위해서, 다운로드 받은 dataset으로부터  `train.db` 파일을 생성합니다.
또한 word의 인코딩을 위한 vocabulary 생성합니다. (동시에 word 분절을 위한 spm.model도 생성됩니다.)

추가로 제출을 위한 `dev.db`, `test.db` 파일도 생성합니다.

#### 1. `train.db`, `dev.db`, `test.db` 생성하기
```bash
python preprocess.py make_db train
python preprocess.py make_db dev
python preprocess.py make_db test
```
data/ 폴더내에 `train.db`가 생성됩니다.  동시에 data/img/train 폴더가 생성되며, 이 폴더 내에 [PID].pt가 개별로 저장됩니다. 예를들면 `data/img/train/H2829766805.pt` 파일이 생성됩니다. 이렇게 따로 저장하는 이유는 학습 시 빠르게 불러오기 위해서입니다. 

#### 2. Vocabulary 생성하기
```bash
python preprocess.py build_vocab train
```
나중에 더 자세히 언급하도록 수정하겠지만 중간에 생성되는 파일 `data/vocab/train_title.txt`은 spm.model 파일을 생성한 후에는 더 이상 필요없습니다. spm.model의 크기는 매우 작습니다.

### 단계 3: 학습하기
```bash
python train.py -j 12 -b 2048 --hidden_size 700 --prefix h700_d0.3_ 
```

#### 모델의 크기 줄이기
학습시에는 모델을 checkpoint로 활용하기 위해서 optimizer의 파라미터도 함께 저장합니다. 그러나 실제 추론 과정에선 필요 없으므로 optimizer의 파라미터는 제거할 수 있습니다. 
```bash
python utils/remove_opt_params.py --model ../output/best_h700_d0.3_it2vec.pth.tar
```

### 단계 4: 추론하기
여러 모델들을 앙상블하여 추론하였습니다. 현재 6개의 앙상블로 모델이 구성되어 있습니다.
싱글 모델용 추론 파일 추가 예정
```bash
python inference.py -j 12 -b 2048 --resume output/best_h1000_d0.3_it2vec.pth.tar --div dev
```
싱글 모델: 1.069~1.075
4개 앙상블 모델: 1.082713
6개 앙상블 모델: 1.084474

아래의 링크에서 6개의 pre-trained 모델 파일을 다운로드하여 output/ 폴더에 넣어준 후에 위의 명령어를 실행하면 됩니다.

* [best_h700_d0.3_it2vec.pth.tar](https://www.dropbox.com/s/aa1vk61c0pphlj9/best_h700_d0.3_it2vec.pth.tar?dl=0)
* [best_h800_d0.3_it2vec.pth.tar](https://www.dropbox.com/s/f76otyd2nfnhlwp/best_h800_d0.3_it2vec.pth.tar?dl=0)
* [best_h900_d0.3_it2vec.pth.tar](https://www.dropbox.com/s/785ktor5dwzoxjt/best_h900_d0.3_it2vec.pth.tar?dl=0)
* [best_h1000_d0.3_it2vec.pth.tar](https://www.dropbox.com/s/s5whaim2lo4zljd/best_h1000_d0.3_it2vec.pth.tar?dl=0)
* [best_h1100_d0.3_it2vec.pth.tar](https://www.dropbox.com/s/pdz7r20kjy0syyp/best_h1100_d0.3_it2vec.pth.tar?dl=0)
* [best_h1200_d0.3_it2vec.pth.tar](https://www.dropbox.com/s/eoch17ody1bhod4/best_h1200_d0.3_it2vec.pth.tar?dl=0)

위 모델들의 총 크기는 600MB정도입니다.
