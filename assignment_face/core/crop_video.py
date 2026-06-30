def sample_recording_frames(
    frames: list[dict],
    duration_seconds: int = 10,
    frames_per_second: int = 5,
) -> list[dict]:

    if not frames:
        return []

    sampled = []
    for second_index in range(duration_seconds):
        start = float(second_index)
        end = float(second_index + 1)
        bucket = [item for item in frames if start <= float(item["timestamp"]) < end]
        if not bucket:
            continue

        max_items = min(frames_per_second, len(bucket))
        if max_items == 1:
            positions = [0]
        else:
            last_index = len(bucket) - 1
            positions = [round(i * last_index / (max_items - 1)) for i in range(max_items)]
        for position in positions:
            sampled.append(bucket[position])

    return sampled
