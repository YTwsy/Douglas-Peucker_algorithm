import numpy as np
import matplotlib.pyplot as plt

f = np.empty(shape=(0, 2))


def txt_to_points(filename):

    CPoints = np.loadtxt(filename, delimiter=",", skiprows=0, unpack=False,
                         usecols=(0, 1))

    print(CPoints)
    print('--------------')
    return CPoints

def continus_find(num_list):  #列表中连续数字段寻找
    num_list.sort()
    s = 1
    find_list = []
    have_list = []
    while s <= len(num_list) - 1:
        if num_list[s] - num_list[s - 1] == 1:
            flag = s - 1
            while num_list[s] - num_list[s - 1] == 1:
                s += 1
            find_list.append(num_list[flag:s])
            have_list += num_list[flag:s]
        else:
            s += 1
    return find_list


def CharaFilter(CPoints,Dmax):
    global f
    if(CPoints.shape[0]<3):
        print(CPoints)


        f=np.append(f,CPoints,axis=0)

        return CPoints

    x1=CPoints[0][0]
    y1=CPoints[0][1]

    # CPoints[-1]
    x2=CPoints[-1][0]
    y2=CPoints[-1][1]

    x_dis=x2-x1
    y_dis=y2-y1

    point_dmax_index=-1
    point_dmax_index_list=list()
    dis_max=0

    for i in range(CPoints.shape[0]-2):

        x_cur = CPoints[i+1][0]
        y_cur = CPoints[i+1][1]

        if(x_dis==0):
            A=1
            B=0
            C=-x1
            dis=(A*x_cur+C)/((A**2+B**2)**(1/2))

        else:
            K=y_dis/x_dis
            H=y1-K*x1
            dis=((-K)*x_cur+y_cur-H)/((K**2+1)**(1/2))

        if(abs(dis)>=Dmax and abs(dis)>=dis_max):
            point_dmax_index = i + 1
            if(abs(dis)>dis_max):
                point_dmax_index_list.clear()


            if(dis==dis_max):
                point_dmax_index_list.append(i+1)

            if(dis_max==0):    #初次达到条件
                point_dmax_index_list.append(i+1)

            if(dis==(-dis_max)):
                point_dmax_index_list.append(-(i+1))



    if(point_dmax_index==-1):
        # return  np.vstack((CPoints[0],CPoints[-1]))        #去掉中间点
        CharaFilter(np.vstack((CPoints[0],CPoints[-1])), Dmax)
        return

    if (len(point_dmax_index_list)==1):
        # return np.split(CPoints,[point_dmax_index],axis=0)   #CPoints分成两端
        one_part=np.split(CPoints, [point_dmax_index], axis=0)[0]
        two_part=np.split(CPoints, [point_dmax_index], axis=0)[1]
        CharaFilter(one_part, Dmax)
        CharaFilter(two_part, Dmax)
        return

    if(len(point_dmax_index_list)>=1):  #[2,3,4,6,7,-8,-9,-10]   CPoints分成多端

        max_dis_list=continus_find(point_dmax_index_list)

        #max_dis_list=[[1, 2], [4, 5, 6, 7],[11,12],[-13,-14,-15]]

        max_dis_list_spilt=list()
        for j in range(len(max_dis_list)):
            if(max_dis_list[i][0]>0):

                max_dis_list_spilt.append(max_dis_list[i][0])
                max_dis_list_spilt.append(max_dis_list[i][-1]+1)

            else:
                for k in range(len(max_dis_list)):
                    max_dis_list[k]=-(max_dis_list[k])

                max_dis_list_spilt.append(max_dis_list[i][0])
                max_dis_list_spilt.append(max_dis_list[i][-1]+1)


        # return np.split(CPoints,max_dis_list_spilt)
        for i in np.split(CPoints,max_dis_list_spilt):
            CharaFilter(i,Dmax)

        return

def show_douglas():
    CharaFilter(txt_to_points("line_points.txt"),20)
    print('--------------')
    pre_dou=txt_to_points("line_points.txt")
    plt.plot(pre_dou[:,0], pre_dou[:,1], label='pre_dou', color='g', linewidth=1,linestyle=':')  # 添加linewidth设置线条大小
    # plt.plot(x, list2, label='list2', color='b', linewidth=5)


    plt.grid()  # 添加网格
    print(f)
    plt.plot(f[:,0], f[:,1], label='after_dou', color='r', linewidth=2,linestyle='-')  # 添加linewidth设置线条大小
    plt.show()

show_douglas()
