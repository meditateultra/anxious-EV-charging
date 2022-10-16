from src.utils import SampleFromNormalDistribution
import pandas as pd
import numpy as np

# 单位：美分/千瓦时
# 0点~23点
hour_price = [20, 19, 16, 18, 20, 22, 25, 27, 33, 35, 37, 45, 38, 35, 34, 33, 33, 32, 28, 25, 23, 22, 22, 20]


def get_price(t):
    index = (t + 24 * 10) % 24
    price = hour_price[index] + SampleFromNormalDistribution(0, 1, 1, -5, 5)
    return price


save_path = '..\\price\\trainPrice.xlsx'
h = 1
write_list = []
for i in range(4632):
    insert_list = [h, get_price(h)]
    write_list.append(insert_list)
    h += 1
    if h > 24:
        h -= 24
s = pd.DataFrame(np.array(write_list))
s.to_excel(save_path, sheet_name="Sheet1", index=False)