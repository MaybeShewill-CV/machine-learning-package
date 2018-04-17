data = []
with open('logReg.txt', 'r') as file:
    for line in file:
        data.append(line[:-1].strip(' ').split())


with open('fuck.txt', 'w') as file:
    for info in data:
        info_str = ' '.join(info)
        file.write(info_str + '\n')
