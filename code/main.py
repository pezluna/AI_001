import os
import sys
import random

import logging

from log_conf import *
from load_files import *
from learn import *
from evaluate import *
from flow import *

init_logger()
logger = logging.getLogger("logger")

if __name__ == "__main__":
    debug = False

    if len(sys.argv) > 1:
        if sys.argv[1] == "-d":
            debug = True
            logger.debug(f"Debug mode")

    # pcap 로드
    pcaps_by_folder = []

    for folder in os.listdir("../Zigbee/"):
        if os.path.isdir("../Zigbee/" + folder + "/"):
            pcaps_by_folder.append(load_files("../Zigbee/" + folder + "/"))

    pcaps = []
    for pcaps_in_folder in pcaps_by_folder:
        for pcap in pcaps_in_folder:
            pcaps.append(pcap)

    logger.info(f"Loaded {len(pcaps)} pcaps.")

    if debug:
        logger.debug(f"Debug mode. Using only 1/3 of pcpas.")
        pcaps = random.sample(pcaps, len(pcaps) // 3)

    # flow 생성
    flows = Flows()
    for pcap in pcaps:
        for pkt in pcap:
            flow_key = FlowKey()
            if not flow_key.set_key(pkt):
                continue

            flow_value = FlowValue()
            flow_value.set_raw_value(pkt, flow_key)

            key = flows.find(flow_key)

            if key is None:
                flows.create(flow_key, flow_value, True)
            else:
                flows.append(key[0], flow_value, key[1])

    logger.info(f"Created {len(flows.value)} flows.")

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
    
    flows.sort()
    flows.tune()
    test_flows.sort()
    test_flows.tune()

    logger.info(f"Split {len(flows.value)} flows into {len(test_flows.value)} test flows.")

    # label 데이터 불러오기
    labels = load_lables("../labels/testbed.csv")

    logger.info(f"Loaded {len(labels)} labels.")

    # SVM 모델 생성
    logger.info(f"Creating SVM models...")

    name_svm_model = classify_using_svm(flows, labels, "name")
    dtype_svm_model = classify_using_svm(flows, labels, "dtype")
    vendor_svm_model = classify_using_svm(flows, labels, "vendor")

    logger.info(f"Created SVM models.")

    # RF 모델 생성
    logger.info(f"Creating RF models...")

    name_rf_model = classify_using_random_forest(flows, labels, "name")
    dtype_rf_model = classify_using_random_forest(flows, labels, "dtype")
    vendor_rf_model = classify_using_random_forest(flows, labels, "vendor")

    logger.info(f"Created RF models.")

    # SVM 모델 평가
    logger.info(f"Evaluating models...")

    name_svm_pred = evaluate(name_svm_model, test_flows, labels, "name", "svm")
    dtype_svm_pred = evaluate(dtype_svm_model, test_flows, labels, "dtype", "svm")
    vendor_svm_pred = evaluate(vendor_svm_model, test_flows, labels, "vendor", "svm")

    name_rf_model = evaluate(name_rf_model, test_flows, labels, "name", "rf")
    dtype_rf_model = evaluate(dtype_rf_model, test_flows, labels, "dtype", "rf")
    vendor_rf_model = evaluate(vendor_rf_model, test_flows, labels, "vendor", "rf")

    logger.info(f"Evaluated models.")

    # 결과 저장
    logger.info(f"Saving results...")

    logger.info(f"Saved results.")

    logger.info(f"Done.")