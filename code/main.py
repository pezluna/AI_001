import os
import time

from load_files import *
from process import *

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
flow_raw = {}

for pcap in pcaps:
    for pkt in pcap:
        five_tuple = get_tuple(pkt)

        if five_tuple in flow_raw:
            flow_raw[five_tuple].append(pkt)
        else:
            flow_raw[five_tuple] = [pkt]

print(f"[{time.time() - start_time}] Created {len(flow_raw)} flows.")

print(flow_raw)