from src.logUtils import LogEntry
import os
import pandas as pd


def get_files(dir: str):
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".pkl"):
                yield os.path.join(root, file)


if __name__ == "__main__":
    dir_path = "log"
    df_data = {
        "实例名称": list(),
        "实例内容": list(),
        "输入变量个数": list(),
        "子句个数": list(),
        "子句平均长度": list(),
        "输出变量个数": list(),
        "输出电路规模": list(),
        "Preprocess时间": list(),
        "LearnSkf时间": list(),
        "Refine时间": list(),
        "修复次数": list(),
        "决策树得分": list(),
        "合理采样": list(),
        "总采样": list(),
        "退出阶段": list(),
        "有解": list(),
        "超时": list(),
        "备注": list(),
    }

    for file in get_files(dir_path):
        print("Now: ", file)
        try:
            log_obj = LogEntry.from_file(file)
        except BaseException:
            print("Error: ", file)
            continue

        # score
        file_path = dir_path + "/" + log_obj.instance_name + "_score.txt"
        file = ""
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                file = f.read()

            lines = file.splitlines()
            score_sum = 0
            for line in lines:
                tokens = line.split(" ")
                score = float(tokens[-1])
                score_sum += score
        
            score_mean = score_sum / len(lines)
            df_data["决策树得分"].append(score_mean)
        else:
            df_data["决策树得分"].append(-1)

        df_data["实例名称"].append(log_obj.instance_name)
        df_data["实例内容"].append(log_obj.instance_str)
        df_data["输入变量个数"].append(len(log_obj.input_vars))
        df_data["子句个数"].append(len(log_obj.get_clause_list()))
        df_data["子句平均长度"].append(log_obj.get_clause_list_avg_len())
        df_data["输出变量个数"].append(len(log_obj.output_vars))
        df_data["输出电路规模"].append(log_obj.caculate_circuit_size())
        df_data["Preprocess时间"].append(log_obj.preprocess_time)
        df_data["LearnSkf时间"].append(log_obj.leanskf_time)
        df_data["Refine时间"].append(log_obj.refine_time)
        df_data["修复次数"].append(log_obj.repair_count)
        df_data["合理采样"].append(log_obj.get_samples_acc()[0])
        df_data["总采样"].append(log_obj.num_samples)
        if log_obj.exit_after_preprocess:
            df_data["退出阶段"].append("Preprocess")
        elif log_obj.exit_after_leanskf:
            df_data["退出阶段"].append("LearnSkf")
        elif log_obj.exit_after_refine:
            df_data["退出阶段"].append("Refine")
        else:
            df_data["退出阶段"].append("未知")

        if log_obj.sat:
            df_data["有解"].append("True")
        else:
            df_data["有解"].append("False")

        if log_obj.exit_after_timeout:
            df_data["超时"].append("True")
        else:
            df_data["超时"].append("False")

        if log_obj.exit_after_expection:
            df_data["备注"].append("断言异常退出")
        elif log_obj.exit_after_error:
            df_data["备注"].append("外部程序错误")
        else:
            df_data["备注"].append("无错误")

        df = pd.DataFrame(df_data)
        df.to_excel("log.xlsx")
        df.to_csv("log.csv")
