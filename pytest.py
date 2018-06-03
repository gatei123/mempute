#-*- coding: utf-8 -*-
import mempute as mp
import numpy as np
import  sys
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import pickle
from operator import eq

def get_handle(win_exec):

    drv = mp.driver(3)

    if win_exec == 1:
        con = mp.connect(drv, "loopback", "", "")
    else:
        con = mp.connect(drv, "localhost", "", "")

    stmt = mp.statement(con)

    return stmt

def reshape_list(alist, init_stride, stride, width, margin, limit_row):
    #width -  입력값 또는 목표값 배열의 시퀀스 갯수
    #margin - 입력값과 목표값 배열을 구성하는데 있어 두 배열의 순차가 하나의 입력배열에서 공유되는 경우이면 입력값 배열 구성일 경우 입력값 배열의 마지막으로부터 
    #           목표값 배열의 마지막까지의 입력 배열상의 시퀀스 차이 갯수, 두 배열이 각기 독립적인 입력 인덱스 체게를 갖는 입력 배열로부터 이면 0
    #limit_row - 입력 배열 제한 갯수, 0부터 시작되므로 순차가 공유되는 경우 이 로우 순차 값 하나 전 로우까지 목표값 배열로 채워지고 입력값은 이 하나전 개수에서 margin값을 제하고난 로우수까지 
    #           입력배열에 채워진다. 두 배열이 각기 독립적인 입력 인덱스 체계를 갖는 경우 각각 배열에서 이 로수 인덱스 이전까지 적재된다.
    n_input_row = len(alist)
    sub_input = alist[0]
    n_input_col = len(sub_input)

    if limit_row > 0:
        n_input_row = limit_row

    input_last = n_input_row - (margin + width)
    init_stride += ((input_last - init_stride) % stride)

    stride_origin = init_stride
    output_list = [];
    
    while stride_origin <= input_last:
        in_row = stride_origin;
        nwidth = in_row + width
        out_col = 0
        sub_output = []
        while in_row < nwidth:
            sub_input = alist[in_row]
            #print(sub_input)
            sub_output[out_col:out_col + n_input_col] = sub_input[0:]
            out_col += n_input_col
            in_row += 1
        output_list.append(sub_output);
        stride_origin += stride
    #print("----------------------------------------")
    #print(output_list)

    return output_list

def list_input(stmt, fname, init_stride, stride, width, margin):
    
    f = open(fname, "r")
    str = "";
    sub = []
    while True:
        lines = f.readlines()
        if not lines :
            break

        nest = []

        for i in lines:
            row = i.replace('\n', '').replace('?', '').replace('!', '').replace('.', '').replace(';', '').replace('\'', '\\\'').split(',')
            str = row[0];
            sub = str.split()
            nest.append(sub)
        
        if stride > 0:
            nest = reshape_list(nest, init_stride, stride, width, margin, 0)

        mp.inlist(stmt, nest, 0)

    f.close()

    empt = []
    mp.inlist(stmt, empt, 1)

def reshape_array(d_type, aarray, init_stride, stride, width, margin, limit_row):
    #width -  입력값 또는 목표값 배열의 시퀀스 갯수
    #margin - 입력값과 목표값 배열을 구성하는데 있어 두 배열의 순차가 하나의 입력배열에서 공유되는 경우이면 입력값 배열 구성일 경우 입력값 배열의 마지막으로부터 
    #           목표값 배열의 마지막까지의 입력 배열상의 시퀀스 차이 갯수, 두 배열이 각기 독립적인 입력 인덱스 체게를 갖는 입력 배열로부터 이면 0
    #limit_row - 입력 배열 제한 갯수, 0부터 시작되므로 순차가 공유되는 경우 이 로우 순차 값 하나 전 로우까지 목표값 배열로 채워지고 입력값은 이 하나전 개수에서 margin값을 제하고난 로우수까지 
    #           입력배열에 채워진다. 두 배열이 각기 독립적인 입력 인덱스 체계를 갖는 경우 각각 배열에서 이 로수 인덱스 이전까지 적재된다.
    n_input_row = aarray.shape[0]
    n_input_col = aarray.shape[1]

    if limit_row > 0:
        n_input_row = limit_row

    n_output_col = n_input_col * width

    input_last = n_input_row - (margin + width)
    init_stride += ((input_last - init_stride) % stride)

    n_output_row = ((input_last - init_stride) / stride) +1

    output_arr = np.zeros((int(n_output_row), n_output_col), dtype = d_type)
    
    """
    out_row = 0
    stride_origin = init_stride
    while stride_origin < n_input_row:
        out_col = 0
        in_row = stride_origin
        nwidth = in_row + width
        while in_row < nwidth:
            in_col = 0
            while in_col < n_input_col:
                output_arr[out_row, out_col] = aarray[in_row, in_col]
                out_col += 1
                in_col += 1
            in_row += 1
        out_row += 1
        stride_origin += stride
    """
    out_row = 0#out_row = int(n_output_row) -1
    stride_origin = init_stride
    while stride_origin <= input_last:
        in_row = stride_origin;
        nwidth = in_row + width
        out_col = 0
        while in_row < nwidth:
            output_arr[out_row, out_col:out_col + n_input_col] = aarray[in_row,:]
            out_col += n_input_col
            in_row += 1
        out_row += 1#out_row -= 1
        stride_origin += stride
    """
    print("------------------------------------")
    i = 0
    while i < n_output_row:
        j = 0
        while j < n_output_col:
            b = output_arr[i, j]
            print('%f ' % b, end='')
            j = j + 1
        print("\n")
        i = i + 1
    """
    return output_arr


def array_input(d_type, stmt, fname, init_stride, stride, width, margin):
    #empt = np.zeros((0, 0), dtype = "d")
    f = open(fname, "r")
    
    while True:
        lines = f.readlines()

        if not lines :
            break

        nest = []

        for i in lines:
            row = i.split()
            nest.append(row)

        input_arr = np.array(nest)
        output_arr = reshape_array(d_type, input_arr, init_stride, stride, width, margin, 0)

        mp.inarray(stmt, output_arr, 0)
    
    f.close()

    mp.inarray(stmt, None, 1)


def reshape_array_eval(d_type, aarray, apredic, width, stride, margin):
#apredic - aarray에서 입력 로우 시작 인덱스(에측 시작 로우 인데스 - 예측값이 width가 있는 경우 그 시작 인덱스)
#width - aarray배열에서 로우(시퀀스) 갯수, 로우단위 걸음 간격 수, 
#margin - 평가 단계에서 입력값 배열을 구성하는데 있어 학습단계에서 목표값 배열의 순차가 하나의 입력배열에서 공유되는 경우일때 입력값 배열의 마지막으로부터 
#           목표값 배열의 마지막까지의 입력 배열상의 시퀀스 차이 갯수, 두 배열이 각기 독립적인 입력 인덱스 체게를 갖는 입력 배열로부터 이면 0
    n_input_row = aarray.shape[0]
    n_input_col = aarray.shape[1]
    n_output_col = n_input_col * width

    gap = margin -1
    stride_origin = (apredic - (gap + width))
    n_output_row = ((n_input_row - apredic) / stride)  +1

    output_arr = np.zeros((int(n_output_row), n_output_col), dtype = d_type)
    
    out_row = 0
    while out_row < n_output_row:
        out_col = 0
        in_row = stride_origin
        while out_col < n_output_col:
            output_arr[out_row, out_col:out_col + n_input_col] = aarray[in_row, :]
            out_col += n_input_col
            in_row += 1
        stride_origin += stride
        out_row += 1
    """
    print("------------------------------------")
    i = 0
    while i < n_output_row:
        j = 0
        while j < n_output_col:
            b = output_arr[i, j]
            print('%f ' % b, end='')
            j = j + 1
        print("\n")
        i = i + 1
    """
    return output_arr

def reshape_list_eval(alist, apredic, width, stride, margin):
#apredic - aarray에서 입력 로우 시작 인덱스(에측 시작 로우 인데스 - 예측값이 width가 있는 경우 그 시작 인덱스)
#width - aarray배열에서 로우(시퀀스) 갯수, 로우단위 걸음 간격 수, 
#margin - 평가 단계에서 입력값 배열을 구성하는데 있어 학습단계에서 목표값 배열의 순차가 하나의 입력배열에서 공유되는 경우일때 입력값 배열의 마지막으로부터 
#           목표값 배열의 마지막까지의 입력 배열상의 시퀀스 차이 갯수, 두 배열이 각기 독립적인 입력 인덱스 체게를 갖는 입력 배열로부터 이면 0
    n_input_row = len(alist)
    sub_input = alist[0]
    n_input_col = len(sub_input)
    n_output_col = n_input_col * width

    gap = margin -1
    stride_origin = (apredic - (gap + width))
    n_output_row = ((n_input_row - apredic) / stride)  +1
    
    output_list = [];
    out_row = 0
    while out_row < n_output_row:
        out_col = 0
        in_row = stride_origin
        sub_output = []
        while out_col < n_output_col:
            sub_input = alist[in_row]
            #print(sub_input)
            sub_output[out_col:out_col + n_input_col] = sub_input[0:]
            out_col += n_input_col
            in_row += 1
        output_list.append(sub_output);
        stride_origin += stride
        out_row += 1
    #print("----------------------------------------")
    #print(output_list)
    return output_list


def exec_batch(d_type, stmt, sor_obj, apredic, width, stride, margin):

    if d_type == 'n':
        list_proc = 1
        in_list = reshape_list_eval(sor_obj, apredic, width, stride, margin)
        n_input_row = len(in_list)
    else:
        list_proc = 0
        in_arr = reshape_array_eval(d_type, sor_obj, apredic, width, stride, margin)
        n_input_row = in_arr.shape[0]

    rv = mp.array(stmt, "execute mempute('array', 'eval_input 1 1 0 0 0 0')")

    ah = mp.gain(stmt)

    rv = mp.array(stmt, "execute mempute('array', 'eval_output 0 1 0 0 0 0')")

    outlist = [];
    i = 0
    a = []

    rv = mp.focus(ah, "execute mempute('array', 'eval_input')")
    while i < n_input_row:
        if list_proc == 1:
            mp.inlist(stmt, in_list[i], 1)
        else:
            a = in_arr[i, 0:]
            mp.inarray(stmt, a, 1)
        a = mp.mempute(stmt, "execute mempute('predict', 'eval_input', 'eval_output')")
        #print(a)
        outlist.append(a)
        i += 1

    return outlist #3차원 리스트 혹은 2차원 배얼을 원소로 갖는 1차원 리스트 리턴.

def exec_batch2(d_type, stmt, sor_obj, apredic, width, stride, margin):

    if d_type == 'n':
        list_proc = 1
        in_list = reshape_list_eval(sor_obj, apredic, width, stride, margin)
        n_input_row = len(in_list)
    else:
        list_proc = 0
        in_arr = reshape_array_eval(d_type, sor_obj, apredic, width, stride, margin)
        n_input_row = in_arr.shape[0]
    
    rv = mp.array(stmt, "execute mempute('array', 'eval_input 1 1 0 0 0 0')")

    ah = mp.gain(stmt)

    rv = mp.array(stmt, "execute mempute('array', 'eval_output 0 1 0 0 0 0')")

    outlist = [];
    a = []

    rv = mp.focus(ah, "execute mempute('array', 'eval_input')")
    if list_proc == 1:
        mp.inlist(stmt, in_list, 1)
    else:
        mp.inarray(stmt, in_arr, 1)

    a = mp.mempute(stmt, "execute mempute('predict', 'eval_input', 'eval_output')")
    #print(a)
    
    return a #2차원 리스트 혹은 2차원 배열 리턴

def evaluate(d_type, stmt, fname, apredic, width, stride, margin):

    f = open(fname, "r")
    
    while True:
        lines = f.readlines()

        if not lines :
            break

        nest = []

        for i in lines:
            row = i.split()
            nest.append(row)

        a = exec_batch(d_type, stmt, nest, apredic, width, stride, margin)

    f.close()

    return a


if sys.argv[2] is 'f':
    
    # 데이터를 로딩한다.
    stock_file_name = 'D:/c/testd/amazon/AMZN.csv' # 아마존 주가데이터 파일
    encoding = 'euc-kr' # 문자 인코딩
    names = ['Date','Open','High','Low','Close','Adj Close','Volume']
    raw_dataframe = pd.read_csv(stock_file_name, names=names, encoding=encoding) #판다스이용 csv파일 로딩
    raw_dataframe.info() # 데이터 정보 출력

    # raw_dataframe.drop('Date', axis=1, inplace=True) # 시간열을 제거하고 dataframe 재생성하지 않기
    del raw_dataframe['Date'] # 위 줄과 같은 효과

    stock_input = raw_dataframe.values[1:].astype(np.float) # 금액&거래량 문자열을 부동소수점형으로 변환한다
    print("stock_info.shape: %d", stock_input.shape)
    print("stock_info[0]: %d", stock_input[0])
    train_size = int(stock_input.shape[0] * 0.7)
    print("tran size %f", train_size)
    stock_target = stock_input[:, [-2]] # 타켓은 주식 종가이다

    input_data = reshape_array("f", stock_input, 0, 1, 5, 3, train_size)#3 = 8 - 5(width), 5-5 best
    target_data = reshape_array("f", stock_target, 7, 1, 1, 0, train_size)#입력의 시작으로부터 8칸 뒤에 목표값 시작되게.

    stmt = get_handle(sys.argv[1])

    rv = mp.direct(stmt, "execute mempute('perception', 'stock_predic locale percep_loc')")

    rv = mp.direct(stmt, "execute mempute('channel', 1, '{ld ld ld ld ld L6d }')")
    rv = mp.direct(stmt, "execute mempute('channel', 0, '{ld}')")

    rv = mp.direct(stmt, "execute mempute('sequence', 5, 1, 2)")

    rv = mp.array(stmt, "execute mempute('array', 'stock_input locale array_loc 1 1 1 1 0 0')")
    mp.inarray(stmt, input_data, 1)

    rv = mp.array(stmt, "execute mempute('array', 'stock_target locale array_loc 0 1 1 1 0 0')")
    mp.inarray(stmt, target_data, 1)

    rv = mp.array(stmt, "execute mempute('array', 'stock_input')")

    rv = mp.direct(stmt, "execute mempute('load', 'stock_input', 0, -1)")

    rv = mp.array(stmt, "execute mempute('array', 'stock_target')")

    rv = mp.direct(stmt, "execute mempute('load', 'stock_target', 0, -1)")

    #rv = mp.direct(stmt, "execute mempute('display', 1)")

    rv = mp.direct(stmt, "execute mempute('cognit', 'stock_input', 'stock_target')")

    rv = mp.direct(stmt, "execute mempute('oblivion', 2)")
    
elif eq(sys.argv[2], "f2"):

    stock_file_name = 'D:/c/testd/amazon/AMZN.csv' # 아마존 주가데이터 파일
    encoding = 'euc-kr' # 문자 인코딩
    names = ['Date','Open','High','Low','Close','Adj Close','Volume']
    raw_dataframe = pd.read_csv(stock_file_name, names=names, encoding=encoding) #판다스이용 csv파일 로딩
    del raw_dataframe['Date'] # 위 줄과 같은 효과
    stock_input = raw_dataframe.values[1:].astype(np.float) # 금액&거래량 문자열을 부동소수점형으로 변환한다
    stock_target = stock_input[:, [-2]] # 타켓은 주식 종가이다
    train_size = int(stock_target.shape[0] * 0.7)
    correct_v = np.array(stock_target[train_size:len(stock_target)])
    
    print("---------correct v-----------------")
    n = correct_v.shape[0]
    i = 0
    while i < n:
        data = "%d: %f\n" % (i, correct_v[i])
        print(data)
        i += 1

    stmt = get_handle(sys.argv[1])
    print("-------------------------------------")
    rv = mp.direct(stmt, "execute mempute('perception', 'stock_predic')")
    #rv = mp.direct(stmt, "execute mempute('display', 1)")
    
    r = exec_batch2("f", stmt, stock_input, train_size, 5, 1, 3) #3개 앞 보기

    predict_v = r[:-1, [0]] #끝에 한개 자름.
    
    print("---------correct predict v-----------------")
    n = predict_v.shape[0]
    i = 0
    while i < n:
        data = "%d: %f %f\n" % (i, correct_v[i], predict_v[i])
        print(data)
        i += 1
    
    x = np.concatenate((correct_v, predict_v), axis=1)
    with open(sys.argv[3], 'wb') as sdata:
        pickle.dump(x, sdata)

    plt.figure(1)
    plt.plot(correct_v, 'r')
    plt.plot(predict_v, 'b')
    plt.show()
    
    print("correct red, predict blue")

elif eq(sys.argv[2], "load"):

    with open(sys.argv[3], 'rb') as sdata:
        x = pickle.load(sdata)

    correct_v = x[:, 0]
    predict_v = x[:, 1]

    print("---------correct v-----------------")
    print(correct_v)
    print("---------predict v-----------------")
    print(predict_v)

    plt.figure(7)
    plt.plot(correct_v, 'r')
    plt.plot(predict_v, 'b')
    plt.show()
    
    print("correct red, predict blue")

elif sys.argv[2] is 'g':
    
    x = np.arange(0, 2000, 1)
    z = np.sin(x)

    n = z.shape[0]

    y = np.zeros((int(n), 1), dtype = 'f')

    i = 0
    while i < n:
        y[i, 0] = z[i] + 1.0
        print('%f ' % y[i, 0])
        i += 1
    #plt.plot(z)
    #plt.show()
    
    train_size = int(n * 0.7)
    print("tran size %f", train_size)
    input_data = reshape_array("f", y, 0, 1, 14, 16, train_size)#아래 ㄱㄱ으로 해야함.
    target_data = reshape_array("f", y, 29, 1, 1, 0, train_size)#입력의 시작으로부터 30개 뒤에 목표값 시작되게.

    stmt = get_handle(sys.argv[1])

    rv = mp.direct(stmt, "execute mempute('perception', 'sign_predic locale percep_loc')")

    rv = mp.direct(stmt, "execute mempute('channel', 1, '{ld}')")
    rv = mp.direct(stmt, "execute mempute('channel', 0, '{ld}')")

    rv = mp.direct(stmt, "execute mempute('sequence', 14, 1, 2)")

    rv = mp.array(stmt, "execute mempute('array', 'sign_input locale array_loc 1 1 1 1 0 0')")
    mp.inarray(stmt, input_data, 1)

    rv = mp.array(stmt, "execute mempute('array', 'sign_target locale array_loc 0 1 1 1 0 0')")
    mp.inarray(stmt, target_data, 1)

    rv = mp.array(stmt, "execute mempute('array', 'sign_input')")

    rv = mp.direct(stmt, "execute mempute('load', 'sign_input', 0, -1)")

    rv = mp.array(stmt, "execute mempute('array', 'sign_target')")

    rv = mp.direct(stmt, "execute mempute('load', 'sign_target', 0, -1)")

    rv = mp.direct(stmt, "execute mempute('cognit', 'sign_input', 'sign_target')")

    rv = mp.direct(stmt, "execute mempute('oblivion', 2)")

    correct_v = np.array(y[train_size:len(y)])

    print("---------correct v-----------------")
    n = correct_v.shape[0]
    i = 0
    while i < n:
        data = "%f\n" % correct_v[i]
        print(data)
        i += 1

    r = exec_batch2("f", stmt, y, train_size, 14, 1, 16) # 평가.

    predict_v = r[:-1, [0]] #끝에 한개 자름.
    
    print("---------correct predict v-----------------")
    n = predict_v.shape[0]
    i = 0
    diff = 0
    idiff = 0
    while i < n:
        if correct_v[i] > predict_v[i]:
            if (correct_v[i] - predict_v[i]) > diff:
                diff = correct_v[i] - predict_v[i]
                idiff = i
        else:
            if (predict_v[i] - correct_v[i]) > diff:
                diff = predict_v[i] - correct_v[i]
                idiff = i
        data = "%d: %f %f\n" % (i, correct_v[i], predict_v[i])
        print(data)
        i += 1
    
    msg = "----------- diff i: %d v:%f ------------\n" % (idiff, diff)
    print(msg)

    x = np.concatenate((correct_v, predict_v), axis=1)
    with open(sys.argv[3], 'wb') as sdata:
        pickle.dump(x, sdata)

    plt.figure(1)
    plt.plot(correct_v, 'r')
    plt.plot(predict_v, 'b')
    plt.show()

