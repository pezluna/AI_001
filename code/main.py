import os
import time

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

flows.print("./result_")