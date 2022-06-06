import pandas as pd
import re
from os.path import join


def process(path):
    transfer = "msec_transfer"
    msec = "msec"
    list_ker = ["coo","csr_vector", "csr_scalar" ,"csr_block(2)","csr_block(4)","csr_block(8)","csr_block(16)","csr_block(32)","dia","ell","hyb"]
    f = open(path, 'r')
    file_name = []
    transfer_time = {
            "coo" : [],
            "csr_vector": [],
            "csr_scalar" : [],
            "csr_block(2)" : [],
            "csr_block(4)" : [],
            "csr_block(8)" : [],
            "csr_block(16)" : [],
            "csr_block(32)" : [],
    "ell":[],
    "hyb": [],
    "dia":[]} #kernel : time
    process_time = {
            "coo" : [],
            "csr_vector": [],
            "csr_scalar" : [],
            "csr_block(2)" : [],
            "csr_block(4)" : [],
            "csr_block(8)" : [],
            "csr_block(16)" : [],
            "csr_block(32)" : [],
        "ell": [],
        "hyb": [],
        "dia": []
    } #kernel : time
    lines = f.readlines()
    #print(lines[0])
    for line in lines:
        res = line.split()
        if res[0].startswith("file"):
            r = res[0].split("/")
            r = r[-1].split(".")
            file_name.append(r[0])
        elif res[0].startswith("msec_transfer"):
            kernel = res[1].split("=")
            kernel = kernel[1]
            trans_t = res[0].split("=")#trans time
            trans_t = trans_t[1]
            transfer_time[kernel].append(trans_t)
            p_t = res[-1].split("=")#process time
            p_t = p_t[1]
            process_time[kernel].append(p_t)
        elif res[0].startswith("kernel="):
            kernel = res[0].split("=")
            kernel = kernel[1]
            ins= list_ker.index(kernel)
            trans_t = transfer_time.get(list_ker[ins - 1])
            #print(trans_t[-1])
            transfer_time[kernel].append(trans_t[-1])
            p_t = res[-1].split("=")  # process time
            p_t = p_t[1]
            process_time[kernel].append(p_t)

    f = open(path, 'r')
    lines = f.readlines()
    lens = len(lines)
    for key in ["dia", "ell", "hyb"]:
        transfer_time[key] =  [ -1 for i in range(lens)]
        process_time[key] = [ -1 for i in range(lens)]

    i = -1
    for line in lines:
        res = line.split()
        if res[0].startswith("file"):
            i = i + 1

        kernel = res[1].split("=")
        kernel = kernel[1]
        if kernel in ["dia", "ell", "hyb"]:
            trans_t = res[0].split("=")  # trans time
            trans_t = trans_t[1]
            transfer_time[kernel][i] = trans_t
            p_t = res[-1].split("=")  # process time
            p_t = p_t[1]
            process_time[kernel][i] = p_t
    # for line in lines:
    #     res = line.split()
    #     if res[0].startswith("file"):
    #         i = i + 1
    #         for j in range(0, 11):
    #             res1 = lines[i+j].split()
    #             if res1[1].startswith("msec_transfer"):
    #                 kernel = res1[1].split("=")
    #                 kernel = kernel[1]
    #
    #                 if kernel in ["dia", "ell", "hyb"]:
    #                     trans_t = res1[0].split("=")  # trans time
    #                     trans_t = trans_t[1]
    #                     print(i)
    #                     print(len( transfer_time[kernel]))
    #
    #                     transfer_time[kernel][i] = trans_t
    #                     p_t = res1[-1].split("=")  # process time
    #                     p_t = p_t[1]
    #                     process_time[kernel][i] = p_t



    for key in  transfer_time.keys():
        nn = key + ".xlsx"
        excel_name = join("./",nn)
        pd_file = pd.DataFrame(file_name, columns = ["filename"])
        pd_t1 = pd.DataFrame(transfer_time[key], columns = ["transfer_time"])
        pd_t2 = pd.DataFrame(process_time[key], columns = ["msec"])
        pd_data = pd.DataFrame(columns = ["file_name", "trans_t", "msec"])
        pd_data["file_name"] = pd_file["filename"]
        pd_data["trans_t"] = pd_t1["transfer_time"]
        pd_data["msec"] = pd_t2["msec"]
        pd_data.to_excel(excel_name)











if __name__ == "__main__":
    process(r"benchmark_output.log")
