import json
import pandas as pd

# 替换为你的文件路径
file1 = "/irip/zhounan_2023/Project/LLaVA/playground/data/eval/coco/answers/answers_llava-v1.5-7b_fullft_vfm_filtered_594k.jsonl"
file2 = "/irip/zhounan_2023/Project/LLaVA/playground/data/eval/coco/answers/answers_llava-v1.5-7b_fullft_vfm.jsonl"

# 读取 jsonl 文件
def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

data1 = load_jsonl(file1)
data2 = load_jsonl(file2)

# 构建 DataFrame
df1 = pd.DataFrame(data1).set_index("questionId")[["answer", "gt_answer"]].rename(columns={"answer": "answer_1", "gt_answer": "gt"})
df2 = pd.DataFrame(data2).set_index("questionId")[["answer"]].rename(columns={"answer": "answer_2"})

# 合并两个结果
df = df1.join(df2)

# 清洗 gt_answer 格式（去掉括号）

df["gt"] = df["gt"].str.strip('()')

# 判断是否预测正确
df["correct_1"] = df["answer_1"] == df["gt"]
df["correct_2"] = df["answer_2"] == df["gt"]

# 统计各类情况
both_correct = ((df["correct_1"]) & (df["correct_2"])).sum()
both_wrong   = ((~df["correct_1"]) & (~df["correct_2"])).sum()
only_1_correct = ((df["correct_1"]) & (~df["correct_2"])).sum()
only_2_correct = ((~df["correct_1"]) & (df["correct_2"])).sum()

# 输出结果
print("✅ 两个模型都预测正确的数量:", both_correct)
print("❌ 两个模型都预测错误的数量:", both_wrong)
print("🔶 模型1正确，模型2错误的数量:", only_1_correct)
print("🔷 模型2正确，模型1错误的数量:", only_2_correct)
