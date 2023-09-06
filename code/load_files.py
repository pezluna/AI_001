import os
import pyshark

def load_files(path):
    pcaps = []
    for file in os.listdir(path):
        if file.endswith(".pcapng"):
            pcap = pyshark.FileCapture(path + file, include_raw=True, use_json=True)

            pcaps.append(pcap)
    return pcaps

if __name__ == '__main__':
    print('This is not a runnable file.')
    exit(1)