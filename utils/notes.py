import time

start_time = time.perf_counter()
for i in range(100000):
    print(i)
stop_time = time.perf_counter()

print(stop_time-start_time)