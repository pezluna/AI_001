import os
import time
import random

from load_files import *
from process import *
from flow import *

print("Loading files...")

start_time = time.time()

# pcap 로드
pcaps_by_folder = []

for folder in os.listdir("../Zigbee/"):
    if os.path.isdir("../Zigbee/" + folder + "/"):
        pcaps_by_folder.append(load_files("../Zigbee/" + folder + "/"))

pcaps = []
for pcaps_in_folder in pcaps_by_folder:
    for pcap in pcaps_in_folder:
        pcaps.append(pcap)

print(f"[{time.time() - start_time}] Loaded {len(pcaps)} pcaps.")

# flow 생성
flows = Flows()
for pcap in pcaps:
    for pkt in pcap:
        flow_key = FlowKey()
        if not flow_key.set_key(pkt):
            continue
        # 
        flow_value = FlowValue()
        flow_value.set_raw_value(pkt, flow_key)
        # 
        key = flows.find(flow_key)
        # 
        if key is None:
            flows.create(flow_key, flow_value, True)
        else:
            flows.append(key[0], flow_value, key[1])

print(f"[{time.time() - start_time}] Created {len(flows.value)} flows.")

# flow 정렬
flows.sort()

print(f"[{time.time() - start_time}] Sorted flows.")

# flow 튜닝
flows.tune()

print(f"[{time.time() - start_time}] Tuned flows.")

# 각 flow에서 랜덤하게 30%의 패킷을 추출하여 test set으로 분리
# 나머지 70%의 패킷을 train set으로 사용
test_flows = Flows()

for k in flows.value:
    flow = flows.value[k]
    length = len(flow)
    test_length = int(length * 0.3)

    test_flows.value[k] = random.sample(flow, test_length)

    for i in test_flows.value[k]:
        flow.remove(i)

flows.tune()
test_flows.sort()
test_flows.tune()

print(f"[{time.time() - start_time}] Splitted test flows.")

# label 데이터 불러오기
labels = load_lables("../labels/testbed-01.csv")

print(f"[{time.time() - start_time}] Loaded labels.")

# train set과 labels를 이용하여 SVM 학습

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# 각 flows.value에 대해 학습
# 학습 모델은 총 3개로, label[3], label[4], label[5]를 각각 예측
# 모델은 SVM을 사용하며, 각 라벨마다 1개의 모델이 생성되어야 함

X = []
name = []
dtype = []
vendor = []

for k in flows.value:
    flow = flows.value[k]
    for i in range(len(flow)):
        data = flow[i]
        # label 데이터를 이용하여 y에 추가
        # label 데이터는 다음과 같은 형태로 저장되어 있음
        # id, protocol, additional, 'name', 'type', 'vendor'

        for label in labels:
            if label[0] == k.sid or label[0] == k.did:
                if label[1] == k.protocol and label[2] == k.additional:
                    name.append(label[3])
                    dtype.append(label[4])
                    vendor.append(label[5])
                    break
        else:
            print("Error: Label not found")
            exit(1)

        # flow 데이터를 이용하여 X에 추가

        X.append([data.delta_time, data.direction, data.length])

nameX = np.array(X)
dtypeX = np.array(X)
vendorX = np.array(X)
name = np.array(name)
dtype = np.array(dtype)
vendor = np.array(vendor)

# name 학습
print(f"[{time.time() - start_time}] Training name model...")
name_model = svm.SVC()
name_model.fit(nameX, name)

# dtype 학습
# print(f"[{time.time() - start_time}] Training dtype model...")
# dtype_model = svm.SVC()
# dtype_model.fit(dtypeX, dtype)

# vendor 학습
# print(f"[{time.time() - start_time}] Training vendor model...")
# vendor_model = svm.SVC()
# vendor_model.fit(vendorX, vendor)

print(f"[{time.time() - start_time}] Trained models.")

# cpickle을 이용하여 학습한 모델을 저장

import _pickle as pickle

# name 모델 저장
print(f"[{time.time() - start_time}] Saving name model...")
with open('name_model.pkl', 'wb') as f:
    pickle.dump(name_model, f)

# dtype 모델 저장
# print(f"[{time.time() - start_time}] Saving dtype model...")
# with open('dtype_model.pkl', 'wb') as f:
#     pickle.dump(dtype_model, f)

# vendor 모델 저장
# print(f"[{time.time() - start_time}] Saving vendor model...")
# with open('vendor_model.pkl', 'wb') as f:
#     pickle.dump(vendor_model, f)

print(f"[{time.time() - start_time}] Saved models.")

# test set을 이용하여 학습한 모델을 평가

X = []
name = []
dtype = []
vendor = []

for k in test_flows.value:
    for i in test_flows.value[k]:
        # label 데이터를 이용하여 y에 추가
        # label 데이터는 다음과 같은 형태로 저장되어 있음
        # id, protocol, additional, 'name', 'type', 'vendor'

        for label in labels:
            if label[0] == k.sid or label[0] == k.did:
                if label[1] == k.protocol and label[2] == k.additional:
                    name.append(label[3])
                    dtype.append(label[4])
                    vendor.append(label[5])
                    break
        else:
            print("Error: Label not found")
            exit(1)

        # flow 데이터를 이용하여 X에 추가

        X.append([i.delta_time, i.direction, i.length])

X = np.array(X)
name = np.array(name)
dtype = np.array(dtype)
vendor = np.array(vendor)

# name 평가
print(f"[{time.time() - start_time}] Evaluating name model...")
name_pred = name_model.predict(X)
print(f"[{time.time() - start_time}] Accuracy: {accuracy_score(name, name_pred)}")
print(f"[{time.time() - start_time}] Confusion matrix:")
print(confusion_matrix(name, name_pred))

# dtype 평가
# print(f"[{time.time() - start_time}] Evaluating dtype model...")
# dtype_pred = dtype_model.predict(X)
# print(f"[{time.time() - start_time}] Accuracy: {accuracy_score(dtype, dtype_pred)}")
# print(f"[{time.time() - start_time}] Confusion matrix:")
# print(confusion_matrix(dtype, dtype_pred))

# vendor 평가
# print(f"[{time.time() - start_time}] Evaluating vendor model...")
# vendor_pred = vendor_model.predict(X)
# print(f"[{time.time() - start_time}] Accuracy: {accuracy_score(vendor, vendor_pred)}")
# print(f"[{time.time() - start_time}] Confusion matrix:")
# print(confusion_matrix(vendor, vendor_pred))

# print(f"[{time.time() - start_time}] Finished.")