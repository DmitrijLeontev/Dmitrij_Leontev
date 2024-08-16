# counter.py
total_requests = 0

def increment_requests():
    global total_requests
    total_requests += 1

def get_total_requests():
    return total_requests
