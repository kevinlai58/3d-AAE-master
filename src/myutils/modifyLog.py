import re, os
import matplotlib.pyplot as plt
import pandas as pd

def parse_json(filename):
    df = pd.read_csv(filename, sep=' ')
    print(df.head())
    print(df.columns)

    l = list(df.loss)
    print(l)


def create_file_without_line_contain_str(file, str_to_be_deleted):
    with open(file, "r", encoding="utf-8") as f1, open("%s+modify.txt" % file, "w", encoding="utf-8") as f2:
        for line in f1:
            if str_to_be_deleted in line:
                continue
            f2.write(line)


def create_file_with_line_contain_str(file, str):
    with open(file, "r", encoding="utf-8") as f1, open("%s+modify.txt" % file, "w", encoding="utf-8") as f2:
        for line in f1:
            if str in line:
                f2.write(line)


def create_file_without_str(file, str_to_be_deleted):
    with open(file, "r", encoding="utf-8") as f1, open("%s_modify.txt" % file, "w", encoding="utf-8") as f2:
        for line in f1:
            f2.write(re.sub(str_to_be_deleted, "", line))


def make_list_data(file):
    list_loss_d = []
    list_loss_eg = []
    list_lambda = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            row = line.split(" ")
            list_loss_d.append(float(row[4]))
            list_loss_eg.append(float(row[6]))
            list_lambda.append(float(row[11]))
    return list_loss_d, list_loss_eg, list_lambda

def make_list_data2(file):
    list_loss_d = []
    list_loss_eg = []
    list_lambda = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            row = line.split(" ")
            list_loss_d.append(float(row[4]))
            list_loss_eg.append(float(row[6]))
            list_lambda.append(float(row[11]))
    return list_loss_d, list_loss_eg, list_lambda

def make_list_data2(file):
    list_loss_d = []
    list_loss_eg = []
    list_loss_e = []
    list_loss_g = []
    list_loss_gp = []
    list_loss_lambda = []

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            row = line.split(" ")
            list_loss_d.append(float(row[4]))
            list_loss_eg.append(float(row[6]))
            list_loss_e.append(float(row[8]))
            list_loss_g.append(float(row[10]))
            list_loss_gp.append(float(row[12]))
            list_loss_lambda.append(float(row[17]))

    return list_loss_d, list_loss_eg,list_loss_e,list_loss_g,list_loss_gp,list_loss_lambda


def plot_loss(list_of_loss):
    plt.figure()
    left = 200
    length_of_loss = len(list_of_loss[0])-left
    x_axis = [*range(length_of_loss)]

    Data_1, = plt.plot(x_axis, list_of_loss[0][left:], 'r-.^', label='loss_d')  # 畫線
    Data_2, = plt.plot(x_axis, list_of_loss[1][left:], 'g--*', label='loss_eg')  # 畫線
    Data_3, = plt.plot(x_axis, list_of_loss[2][left:], 'b--*', label='lambda')  # 畫線

    plt.tick_params(axis='both', labelsize=12, color='green')
    plt.legend(handles=[Data_1, Data_2, Data_3])



def plot_loss_test(list_of_loss):
    # length_of_loss = len(list_of_loss[0])
    # list = list_of_loss[0]

    length_of_loss = 3000
    list = list_of_loss[0:3000]

    x_axis = [*range(length_of_loss)]
    # 製作figure
    fig = plt.figure()

    # 圖表的設定
    ax = fig.add_subplot(1, 1, 1)

    # 散佈圖
    ax.scatter(x_axis, list, color='red',s=1)
    plt.show()


def main():
    # list_loss_d, list_loss_eg, list_lambda = make_list_data2("bone_lambda10to2.txt")
    # list_of_list = [list_loss_d, list_loss_eg, list_lambda]
    # plot_loss(list_of_list)
    #
    # # with open("lambda10to0.1_loss.txt", "w", encoding="utf-8") as f:
    # #     for loss_d, loss_eg, lambda_ in zip(*list_of_list):
    # #         f.write('Loss_D: %s Loss_EG: %s lambda: %s\n' % (loss_d, loss_eg, lambda_))
    #
    #
    # list_loss_d, list_loss_eg = make_list_data2("lambda2_plt.txt")
    # list_lambda = [2]*3206
    # list_of_list = [list_loss_d, list_loss_eg, list_lambda]
    # plot_loss(list_of_list)
    # plt.show()
    #
    # import pandas as pd
    # data = pd.read_table('log.txt', header=None, encoding='gb2312', sep=',', index_col=0)


    import pandas as pd
    df = pd.read_csv('log.csv',sep='\s+')
    print(df.index)
    c = (df["b"])
    print(c)


    exit()

if __name__ == '__main__':
    # parse_json('bone_lambda10to2.txt')
    # exit()
    main()
