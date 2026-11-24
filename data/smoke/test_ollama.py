
import requests
import json

# Ollama 服务地址
OLLAMA_URL = "http://localhost:11434"

# 修改为本地已经下载的模型名称
MODEL_NAME = "gemma4:31b"


def test_ollama():
    print("=" * 50)
    print("Testing Ollama Service")
    print("=" * 50)

    # 1. 测试服务是否启动
    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/version",
            timeout=5
        )
        response.raise_for_status()
        print("[SUCCESS] Ollama service is running!")
        print("Version:", response.json().get("version"))

    except requests.exceptions.RequestException as e:
        print("[FAILED] Cannot connect to Ollama:", e)
        return

    # 2. 查看本地模型
    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=10
        )
        response.raise_for_status()

        models = response.json().get("models", [])
        model_names = [m["name"] for m in models]

        print("\nAvailable models:")
        for name in model_names:
            print(" -", name)

        if not any(
            name == MODEL_NAME or name.startswith(MODEL_NAME + ":")
            for name in model_names
        ):
            print(f"\n[FAILED] Model {MODEL_NAME} not found!")
            return

        print(f"\n[SUCCESS] Model {MODEL_NAME} is available!")

    except requests.exceptions.RequestException as e:
        print("[FAILED] Cannot retrieve models:", e)
        return

    # 3. 测试模型推理
    print("\nTesting model inference...")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": "Hello! Please introduce yourself briefly."
            }
        ],
        "stream": False
    }

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=300
        )
        response.raise_for_status()

        result = response.json()
        answer = result["message"]["content"]

        print("\n[SUCCESS] Inference completed!")
        print("-" * 50)
        print(answer)
        print("-" * 50)

        print("\nTotal duration:",
              result.get("total_duration", 0) / 1e9, "seconds")

    except requests.exceptions.RequestException as e:
        print("[FAILED] Inference request failed:", e)
    except (KeyError, ValueError) as e:
        print("[FAILED] Invalid response:", e)


if __name__ == "__main__":
    test_ollama()
