# ToDo
## Not Started
### 1. flow 객체의 key로 쓰일 tuple 생성 (~9/11)
  * 외부 데이터셋 활용을 위한 TCP/UDP 확장 방안 도출
    * 프로토콜 확장 가능성을 고려하여 수정이 용이하도록 코드를 작성하였음
      * 그러나 pcapng 파일에서 튜플을 추출하는 과정과 csv 파일에서 튜플을 추출하는 과정이 상이하기에 검증 필요
        * 대상 데이터셋 후보: Bot-IoT

### 2. 모델 학습 방법 수립 (~9/14)
  * *학습에 사용할 모델 선택(DT, RF, SVM, RNN, LSTM, etc.)
    *  현재 고려 중인 모델: RF, SVM, RNN의 3종
    * 학습 모델 구조 설계(epoch, batch size, hidden layer, etc.)

### 3. 검토 및 검증 (~9/15)
  * 모델 학습 및 평가

## In Progress
### 1. 모듈화 (~9/8)
  * 모듈화되지 않은 레거시 코드에서 잦은 에러 발생
  * 개발 및 디버깅의 편의성을 위한 코드 재구성 및 모듈화 작업 수행
	
### 2. flow 객체의 value(모델의 feature)로 쓰일 tuple 생성 (~9/8)
  * 해당 tuple에 암호화된 payload에 대한 정보를 넣을지에 대한 고민
    * 기기의 종류 및 기기의 행동에 따라 payload의 형태가 결정된다는 가정
    * 해당 가정이 참일 경우, payload에 대한 정보 역시 분류 작업에 있어 하나의 특징으로 활용될 수 있다는 판단 도출
    * payload의 어떤 속성이 분류에 유효한지 파악 필요

### 3. 모델 학습 방법 수립 (~9/8)
  * 학습 데이터와 테스트 데이터, 검증 데이터를 나누는 방법(비율 등)

## Done
### 1. 데이터 수집 및 증강
  * 데이터 수집(Samsung SmartThings Hub, Aqara Smart Hub)
    * 두 허브와 연결된 각 단말 간 Zigbee 통신 패킷 확보
  * 데이터 증강 함수 구현
    * timestamp에 난수 배수를 적용
    * seq_num을 조정
  * 최종 데이터셋
    * 각 허브마다 3시간/79개의 pcapng 파일을 확보
	
### 2. flow 객체의 key로 쓰일 tuple 생성
  * 패킷의 상위 레이어에 따른 내부 속성 파악
    * ZBEE_NWK_RAW: Zigbee 통신 패킷
    * WPAN_RAW: IEEE 802.15.4 패킷 (광고 패킷으로 추정)
    * DATA_RAW: IEEE 802.15.4 패킷 (현재 용도 식별 실패)
  * 파악한 속성 중 아래의 정보를 가리키는 속성 식별
    * Source Node ID
    * Destination Node ID (광고일 경우, 0xFFFF로 설정)
    * PAN ID
  * 패킷 수집 시 같이 획득한 채널을 포함하여, 최종적으로 아래의 5가지 요소를 지닌 tuple을 추출
    * Source Node ID
    * Destination Node ID
    * Protocol
    * PAN ID
    * Channel

### 3. flow 객체의 value(모델의 feature)로 쓰일 tuple 생성
  * 같은 flow로 분류되는 모든 패킷들에 대하여 아래의 값을 획득할 수 있는 함수 작성
    * timestamp 증가량
	* length
	* direction(정방향/역방향)

### 4. GitHub 연동
  * 개발 및 관리를 위한 Repository 생성