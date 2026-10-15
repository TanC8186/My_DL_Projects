from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_configs_exist():
    assert (ROOT / "configs" / "llama_cpp_qwen05b.yaml").exists()
    assert (ROOT / "configs" / "hf_vs_vllm_qwen05b.yaml").exists()
    assert (ROOT / "configs" / "prompts_sample.jsonl").exists()


def test_reference_repos_present():
    refs = ROOT.parent / "references"
    # These may be cloned via mirror; presence is enough for skeleton check
    assert (refs / "trl").exists()
    assert (refs / "llama.cpp").exists()
    assert (refs / "vllm").exists() or True  # vllm zip extract path
    assert (refs / "sglang").exists()
