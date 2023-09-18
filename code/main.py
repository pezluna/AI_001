import os

import logging

from log_conf import *
from load_files import *
from learn import *
from evaluate import *
from flow import *

init_logger()
logger = logging.getLogger("logger")

if __name__ == "__main__":
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

    logger.info(f"Created {len(test_flows.value)} test flows.")
    
    flows.sort()
    flows.tune()
    test_flows.sort()
    test_flows.tune()

    logger.info(f"Sorted and tuned flows.")

    # label 데이터 불러오기
    labels = load_lables("../labels/testbed.csv")

    logger.info(f"Loaded {len(labels)} labels.")

    # 모델 생성
    model_list = ["ovo", "ovr", "rf"]
    mode_list = ["name", "dtype", "vendor"]

    for model in model_list:
        for mode in mode_list:
            logger.info(f"Creating {mode} {model} model...")
            if model == "rf":
                model = learn(flows, labels, mode, model)
            else:
                model = learn(flows, labels, mode, model)
            logger.info(f"Created {mode} {model} model.")

            globals()[mode + "_" + model + "_model"] = model

    logger.info(f"Created models.")

    # 모델 평가
    logger.info(f"Evaluating models...")
    for model in model_list:
        for mode in mode_list:
            evaluate(test_flows, labels, mode, model, globals()[mode + "_" + model + "_model"])

    logger.info(f"Evaluated models.")
    
    logger.info(f"Done.")