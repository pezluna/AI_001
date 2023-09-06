import pyshark

def get_tuple_zigbee(pkt):
    # 아래 내용 수정 필요
    # 반환 형식은 (five_tuple, four_tuple)
    # five_tuple의 Source Node와 Destination Node는 오름차순으로 정렬하여 반환
    destination_node, source_node = pkt.zbee_nwk.addr

    # 문자열의 오름차순 정렬
    if destination_node > source_node:
        destination_node, source_node = source_node, destination_node

    five_tuple = (
        source_node, 
        destination_node, 
        protocol='Zigbee', 
        A1=pkt.layers[0].dst_pan, 
        A2=0
    )

    return five_tuple

def get_tuple_15d4(pkt):
    # 아래 내용 수정 필요
    # 반환 형식은 (five_tuple, four_tuple)
    # five_tuple의 Source Node와 Destination Node는 오름차순으로 정렬하여 반환

    source_node = pkt.wpan.addr64
    destination_node = pkt.wpan.dst16
    five_tuple = (

    )

    return five_tuple

def get_default_tuple(pkt):
    # 아래 내용 수정 필요
    # 반환 형식은 (five_tuple, four_tuple)
    # five_tuple의 Source Node와 Destination Node는 오름차순으로 정렬하여 반환

    five_tuple = (

    )
    
    return five_tuple

def get_tuple(pcap):
    protocol = {
        "ZBEE_NWK_RAW": get_tuple_zigbee,
        "WPAN_RAW": get_tuple_15d4,
        "default": get_default_tuple
    }

    # 아래 내용 수정 필요
    try:
        return protocol[pcap.highest_layer](pcap)
    except:
        return protocol['default'](pcap)