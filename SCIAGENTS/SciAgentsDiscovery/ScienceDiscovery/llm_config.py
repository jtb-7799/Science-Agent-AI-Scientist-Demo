# import autogen
# from openai import AzureOpenAI

# # -------------------------- 1. 鍒濆鍖?Azure OpenAI 瀹㈡埛绔?--------------------------
# # 鐩存帴瀹氫箟 Azure 閰嶇疆甯搁噺锛堥伩鍏嶉€氳繃 client 瀹炰緥鑾峰彇灞炴€э級
# AZURE_API_KEY = "your-azure-openai-api-key"
# AZURE_API_VERSION = "2024-12-01-preview"
# AZURE_ENDPOINT = "https://your-resource-name.openai.azure.com/"
# AZURE_DEPLOYMENT_4O = "gpt-4o"  # 浣犵殑 GPT-4o 閮ㄧ讲鍚?# AZURE_DEPLOYMENT_4TURBO = "gpt-4o"  # 浣犵殑 GPT-4-turbo 閮ㄧ讲鍚?
# # 鍒濆鍖?Azure OpenAI 瀹㈡埛绔?# client = AzureOpenAI(
#     api_key=AZURE_API_KEY,
#     api_version=AZURE_API_VERSION,
#     azure_endpoint=AZURE_ENDPOINT  # 鍒濆鍖栧弬鏁帮紙涓嶄細浣滀负瀹炰緥灞炴€э級
# )

# # -------------------------- 2. 瀹氫箟 Azure OpenAI 閰嶇疆鍒楄〃 --------------------------
# # 娉ㄦ剰锛欰zure 闇€瑕佹寚瀹?deployment_name锛堝嵆浣犲湪 Azure 涓婇儴缃茬殑妯″瀷鍚嶇О锛?# config_list_azure_4o = [
#     {
#         "model": AZURE_DEPLOYMENT_4O,
#         "api_key": AZURE_API_KEY,
#         "base_url": f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT_4O}",
#         "api_type": "azure",
#         "api_version": AZURE_API_VERSION,
#     }
# ]

# config_list_azure_4turbo = [
#     {
#         "model": AZURE_DEPLOYMENT_4TURBO,
#         "api_key": AZURE_API_KEY,
#         "base_url": f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT_4TURBO}",
#         "api_type": "azure",
#         "api_version": AZURE_API_VERSION,
#     }
# ]

# # -------------------------- 3. 鏇挎崲鍘熸湁鐨?LLM 閰嶇疆 --------------------------
# # 鍘?gpt4o_config 鈫?鏀逛负 Azure 鐗堟湰
# gpt4o_config = {
#     "cache_seed": 42,
#     "temperature": 0.0,
#     "config_list": config_list_azure_4o,  # 鏇挎崲涓?Azure 閰嶇疆
#     "timeout": 540000,
# }

# gpt4o_config_graph = {
#     "cache_seed": 42,
#     "temperature": 0.1,
#     "config_list": config_list_azure_4o,  # 鏇挎崲涓?Azure 閰嶇疆
#     "timeout": 540000,
#     "max_tokens": 2048
# }

# gpt4turbo_config_graph = {
#     "cache_seed": 42,
#     "temperature": 0.2,
#     "config_list": config_list_azure_4turbo,  # 鏇挎崲涓?Azure 閰嶇疆
#     "timeout": 540000,
# }

# gpt4turbo_config = {
#     "cache_seed": 42,
#     "temperature": 0,
#     "config_list": config_list_azure_4turbo,  # 鏇挎崲涓?Azure 閰嶇疆
#     "timeout": 540000,
# }




import autogen

# ===================== AZURE 姝ｇ‘閰嶇疆 =====================
AZURE_API_KEY = "your-azure-openai-api-key"
AZURE_ENDPOINT = "https://your-resource-name.openai.azure.com/"  # 浣犵殑鍦板潃
AZURE_API_VERSION = "2024-12-01-preview"

# 鈿狅笍鈿狅笍鈿狅笍 杩欓噷蹇呴』鏀规垚浣犮€怉zure 閲岀湡姝ｇ殑閮ㄧ讲鍚嶇О銆戔殸锔忊殸锔忊殸锔?DEPLOYMENT_NAME = "gpt-4o"  # 鏀规垚浣犺嚜宸辩殑锛?

config_list = [
    {
        "api_type": "azure",
        "api_version": AZURE_API_VERSION,
        "api_key": AZURE_API_KEY,
        "azure_endpoint": AZURE_ENDPOINT,
        "model": DEPLOYMENT_NAME,
        "base_url": f"{AZURE_ENDPOINT}openai/deployments/{DEPLOYMENT_NAME}",
    }
]

# 浠ヤ笅瀹屽叏淇濈暀浣犲師鏉ョ殑缁撴瀯锛屼笉鐢ㄥ姩
config_list_4o = config_list
config_list_4turbo = config_list

gpt4o_config = {
    "cache_seed": 42,
    "temperature": 0.0,
    "config_list": config_list_4o,
    "timeout": 120,
}

gpt4o_config_graph = {
    "cache_seed": 42,
    "temperature": 0.1,
    "config_list": config_list_4o,
    "timeout": 120,
    "max_tokens": 2048
}

gpt4turbo_config_graph = {
    "cache_seed": 42,
    "temperature": 0.2,
    "config_list": config_list_4turbo,
    "timeout": 120,
}

gpt4turbo_config = {
    "cache_seed": 42,
    "temperature": 0,
    "config_list": config_list_4turbo,
    "timeout": 120,
}
