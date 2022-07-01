import config
os = config.os

# os.system('python ./plot_ssn.py') # 生成数量序列

os.system('python ./gen_i_o.py') 
os.system('python ./main_lstm.py') 
os.system('python ./plot_all.py') 

