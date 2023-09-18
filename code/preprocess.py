import logging

logger = logging.getLogger("logger")

def normalize(value, value_type):
    if value == 0:
        return 0
    
    if value_type == "protocol":
        if "ZBEE_NWK" in value_type:
            return 1
    
    if value_type == "delta_time":
        if value >= 1000:
            return 1
        else:
            return value / 1000
    
    if value_type == "length":
        if value >= 30:
            return 1
        else:
            return value / 30
        
    if value_type == "direction":
        return value
    
    logger.error(f"Cannot normalize {value_type} {value}")
    exit(1)