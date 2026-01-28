import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 你的硅基流动配置
silicon_api_key: str = os.getenv("SILICONFLOW_API_KEY", "")
silicon_api_base = os.getenv("SILICONFLOW_API_BASE", "")

client = OpenAI(api_key=silicon_api_key, base_url=silicon_api_base)

try:
    print("正在查询硅基流动支持的模型列表...")
    models = client.models.list()

    print("\n====== 你的账号可用的模型 ID ======")
    for m in models:
        # 过滤只显示 deepseek 和 bge 相关的，方便查看
        if "deepseek" in m.id.lower() or "bge" in m.id.lower():
            print(f"- {m.id}")

    print("\n==================================")

except Exception as e:
    print(f"查询失败: {e}")
