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

    # 학습용 pcap 로드
    pcaps_by_folder = []

    for folder in os.listdir("../train/"):
        if os.path.isdir("../train/" + folder + "/"):
            pcaps_by_folder.append(load_files("../train/" + folder + "/"))

    train_pcaps = []
    for pcaps_in_folder in pcaps_by_folder:
        for pcap in pcaps_in_folder:
            train_pcaps.append(pcap)

    logger.info(f"Loaded {len(train_pcaps)} pcaps for training.")

    # 테스트용 pcap 로드
    pcaps_by_folder = []

    for folder in os.listdir("../test/"):
        if os.path.isdir("../test/" + folder + "/"):
            pcaps_by_folder.append(load_files("../test/" + folder + "/"))

    test_pcaps = []
    for pcaps_in_folder in pcaps_by_folder:
        for pcap in pcaps_in_folder:
            test_pcaps.append(pcap)

    logger.info(f"Loaded {len(test_pcaps)} pcaps for testing.")

    # flow 생성
    flows = Flows()
    for pcap in train_pcaps:
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

    # test flow 생성
    test_flows = Flows()
    for pcap in test_pcaps:
        for pkt in pcap:
            flow_key = FlowKey()
            if not flow_key.set_key(pkt):
                continue

            flow_value = FlowValue()
            flow_value.set_raw_value(pkt, flow_key)

            key = test_flows.find(flow_key)

            if key is None:
                test_flows.create(flow_key, flow_value, True)
            else:
                test_flows.append(key[0], flow_value, key[1])
    
    flows.sort()
    flows.tune()
    test_flows.sort()
    test_flows.tune()

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