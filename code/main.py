import os
import sys

import logging

from log_conf import *
from load_files import *
from learn import *
from evaluate import *
from flow import *

init_logger()
logger = logging.getLogger("logger")

if __name__ == "__main__":
    logger.info(f"Starting...")

    debug = False
    if len(sys.argv) == 2:
        if sys.argv[1] == "-d":
            logger.info(f"Starting in debug mode...")
            debug = True

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
    logger.info(f"Creating flows...")
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
    logger.info(f"Creating test flows...")
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
    
    # flow 정렬 및 튜닝
    logger.info(f"Sorting and tuning flows...")
    flows.sort()
    flows.tune()
    test_flows.sort()
    test_flows.tune()
    logger.info(f"Sorted and tuned flows.")

    # label 데이터 불러오기
    logger.info(f"Loading labels...")
    labels = load_lables("../labels/testbed.csv")

    logger.info(f"Loaded {len(labels)} labels.")

    if debug:
        logger.debug(f"Labels:")
        for l in labels:
            logger.debug(f"{l}")
        
        logger.debug(f"Flows:")
        for k in flows.value:
            logger.debug(f"{k}")

    # 모델 생성
    model_list = ["ovo",  "rf", "dt"]
    mode_list = ["name", "dtype", "vendor"]

    for model_type in model_list:
        for mode in mode_list:
            logger.info(f"Creating {mode} {model_type} model...")
            if model == "rf":
                model = learn(flows, labels, mode, model)
            logger.info(f"Created {mode} {model_type} model.")

            globals()[mode + "_" + model_type + "_model"] = model

    logger.info(f"Created models.")

    # 모델 평가
    logger.info(f"Evaluating models...")
    for model_type in model_list:
        for mode in mode_list:
            evaluate(test_flows, labels, mode, model_type, globals()[mode + "_" + model_type + "_model"])

    logger.info(f"Evaluated models.")
    
    logger.info(f"Done.")