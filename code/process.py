import pyshark

def get_tuple_zigbee(pkt):
    destination_node, source_node = pkt.zbee_nwk.addr
    if destination_node > source_node:
        destination_node, source_node = source_node, destination_node
    five_tuple = tuple([source_node, destination_node, protocol:='Zigbee', A1:=pkt.layers[0].dst_pan, A2:=0])
    return five_tuple

def get_tuple_15d4(pkt):
    source_node = pkt.wpan.addr64
    destination_node = pkt.wpan.dst16
    five_tuple = tuple([source_node, destination_node, protocol:='IEEE 802.15.4', A1:=0, A2:=0])
    return five_tuple

def get_default_tuple(pkt):
    five_tuple = tuple([pkt.layers[0].src, pkt.layers[0].dst, protocol:=pkt.highest_layer, A1:=0, A2:=0])
    return five_tuple

def get_tuple(pkt):
    protocol = {
        "ZBEE_NWK_RAW": get_tuple_zigbee,
        "WPAN_RAW": get_tuple_15d4,
        "default": get_default_tuple
    }
    # 아래 내용 수정 필요
    try:
        return protocol[pkt.highest_layer](pkt)
    except:
        return protocol['default'](pkt)