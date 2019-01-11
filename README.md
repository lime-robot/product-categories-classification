# 상품 카테고리 분류기

본 분류기는 상품의 타이틀(product)과 이미지 특징(img_feat)만 입력으로 활용하여 대/중/소/상세 카테고리를 예측합니다. 



## Requirements
Ubuntu 16.04, Python 3.6, pytorch 1.0에서 실행을 확인하였습니다.

필요한 패키지는 아래의 명령어로 설치할 수 있습니다.
```bash
pip install -r requirements.txt
```



## Getting Started

### Step 1: 데이터 다운로드
작업 디렉터리(예시:`kakao_arena/`) 하위 디렉터리에 `dataset/` 디렉터리를 생성하고, 카카오 아레나 - 쇼핑몰 상품 카테고리 분류 대회의 데이터를 [다운로드](https://arena.kakao.com/c/1/data) 받습니다.

본 소스코드(product-categories-classification)도 작업 디렉터리 하위에 위치시킵니다.

```
kakao_arena
├── dataset/
│   ├── train.chunk.01
│   ├── train.chunk.02
│   ├── train.chunk.03
│   ├── ...
│   └── test.chunk.01
└── product-categories-classification/
    ├── utils/
    ├── doc/
    ├── train.py
    ├── ...
    └── inference.py
```

### Step 2: 데이터 준비 및 보카 생성
다운로드 받은 dataset으로부터 학습을 위해 필요한 파일을 생성해 냅니다.


#### 1. `train.h5`, `dev.h5`, `test.h5` 생성하기
```bash
python preprocess.py make_db train
python preprocess.py make_db dev
python preprocess.py make_db test
```

아래 처럼 `data/` 디렉터리 내에 3개의 파일이 생성되어 위치하게 됩니다.
```
product-categories-classification/
├── data/
    ├── train.h5
    ├── dev.h5
    └── test.h5
```

#### 2. Vocabulary 생성하기
```bash
python preprocess.py build_vocab train
```

`data/vocab` 디렉터리 내에 word를 index로 치환하기 위해 참조할 vocabulary 파일을 생성해 냅니다. 

그리고 unknown word 문제를 완화시키기 위해서 [bpe](https://github.com/rsennrich/subword-nmt) 방법을 적용할텐데, 이 때 필요한 모델  `data/vocab/spm.model`을 생성합니다. 이 모델을 생성하기 위한 중간파일 `data/vocab/train_title.txt`는 `spm.model` 파일을 생성한 후에는 더 이상 사용되지 않습니다.

### Step 3: 학습하기
```bash
python train.py -j 12 -b 2048 --hidden_size 700 --prefix h700_d0.3_ 
```

#### 모델의 크기 줄이기
학습시에는 모델을 checkpoint로 활용하기 위해서 optimizer의 파라미터도 함께 저장합니다. 그러나 실제 추론 과정에선 필요 없으므로 optimizer의 파라미터는 제거할 수 있습니다. 
```bash
python utils/remove_opt_params.py --model ../output/best_h700_d0.3_it2vec.pth.tar
```

### Step 4: 추론하기
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
