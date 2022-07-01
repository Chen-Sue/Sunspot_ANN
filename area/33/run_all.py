import config
os = config.os

os.system('python ./merge_data.py') # 1874-2021

os.system('python ./gen_i_o.py') # 生成面积序列

os.system('python ./plot_number_area.py') # 数量-面积序列

os.system('python ./gen_i_o_lat_one.py') # 未来一个月

