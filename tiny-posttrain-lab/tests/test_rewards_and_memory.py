from my_lab.memory import estimate_training_vram_gb
from my_lab.rewards import accuracy_reward, extract_final_number


def test_extract_final_number():
    assert extract_final_number("answer is 42") == 42.0
    assert extract_final_number("step1=3 step2=7.5") == 7.5
    assert extract_final_number("no numbers here") is None


def test_accuracy_reward():
    r = accuracy_reward(["The answer is 12"], ["12"])
    assert r == [1.0]
    r = accuracy_reward(["The answer is 12"], ["13"])
    assert r == [0.0]
    r = accuracy_reward(["no number"], ["12"])
    assert r == [0.0]


def test_memory_estimate_increases_with_size():
    small = estimate_training_vram_gb(0.5, lora=True, gradient_checkpointing=True)
    large = estimate_training_vram_gb(7.0, lora=True, gradient_checkpointing=True)
    assert large > small
    full = estimate_training_vram_gb(0.5, lora=False, gradient_checkpointing=False)
    assert full > small
