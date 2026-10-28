import requests
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

from utils.constants import API_URL

TOTAL_REQUESTS = 100
CONCURRENCY = 10


payload = {
    "IRRADIATION": 0.8,
    "AMBIENT_TEMPERATURE": 30,
    "DATE_TIME": "10-10-2026 12:00"
}


def send_request():
    start = time.perf_counter()

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=30
        )

        end = time.perf_counter()

        latency_ms = (end - start) * 1000

        return response.status_code, latency_ms

    except Exception as exc:
        print("Request failed:", exc)
        return None, None


def percentile(values, percentage):
    values = sorted(values)

    index = (len(values) - 1) * (percentage / 100)

    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    weight = index - lower

    return (
        values[lower]
        + (values[upper] - values[lower]) * weight
    )


latencies = []
successful = 0
failed = 0

start_test = time.perf_counter()


with ThreadPoolExecutor(
    max_workers=CONCURRENCY
) as executor:

    futures = [
        executor.submit(send_request)
        for _ in range(TOTAL_REQUESTS)
    ]

    for future in futures:

        status_code, latency = future.result()

        if status_code == 200:

            successful += 1
            latencies.append(latency)

        else:

            failed += 1


end_test = time.perf_counter()


if not latencies:

    print("No successful requests.")
    exit()


p50 = percentile(latencies, 50)
p95 = percentile(latencies, 95)
p99 = percentile(latencies, 99)

total_time = end_test - start_test


print()
print("=" * 50)
print("Cloud Run Load Test")
print("=" * 50)

print(f"Endpoint: {API_URL}")
print(f"Total requests: {TOTAL_REQUESTS}")
print(f"Concurrency: {CONCURRENCY}")

print(f"Successful requests: {successful}")
print(f"Failed requests: {failed}")

print(f"p50 latency: {p50:.2f} ms")
print(f"p95 latency: {p95:.2f} ms")
print(f"p99 latency: {p99:.2f} ms")

print(f"Total test duration: {total_time:.2f} seconds")

print("=" * 50)