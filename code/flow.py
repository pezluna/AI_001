class FlowKey:
    def __init__(self):
        self.sid = None
        self.did = None
        self.protocol = None
        self.additional = None

    def set_key(self, pkt):
        if pkt.highest_layer == 'ZBEE_NWK':
            self.sid = pkt.highest_layer.src
            self.did = pkt.highest_layer.dst
            self.protocol = pkt.transport_layer
            self.additional = pkt.highest_layer.pan_id
            return True
        
        else:
            return False

class FlowValue:
    def __init__(self):
        self.delta_time = None
        self.direction = None
        self.length = None
        self.delta_time = None
    
    def set_raw_value(self, raw_time, direction, length):
        self.raw_time = raw_time
        self.direction = direction
        self.length = length

class Flows:
    def __init__(self):
        self.value = None
    
    def find(self, key):
        for k in self.raw_value:
            if k.protocol == key.protocol and k.additional == key.additional:
                if k.sid == key.sid and k.did == key.did:
                    return k
                elif k.sid == key.did and k.did == key.sid:
                    return k
                
        return None
    
    def create(self, key, value):
        self.value[key] = value
    
    def append(self, key, value):
        self.value[key].append(value)
    
    def sort(self):
        for k in self.value:
            self.value[k].sort(key=lambda x: x.raw_time)

    def tune(self):
        for k in self.value:
            start_time = self.value[k][0].time

            self.value[k][0].delta_time = 0

            for i in range(1, len(self.value[k])):
                self.value.delta_time = self.value[k][i].time - start_time
                start_time = self.value[k][i].time