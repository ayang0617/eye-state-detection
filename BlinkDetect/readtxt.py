def ReadTxt(location):
    txt = []
    temp = ()
    with open(location, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            temp = line.split(' ')
            txt.append(temp)
    return txt



if __name__=='__main__':
    location = "/Users/ayang/PycharmProjects/pythonProject/test.txt"
    txt = ReadTxt(location)
    print(txt)