def project_name(name):
    print(f'Project : {name}')  # Press ⌘F8 to toggle the breakpoint.

def members(list):
    print(f'멤버 : ',end = '')
    for mem in list:
        print(f'{mem}',end=' ')

def where(str):
    print(f'\nin {str}...')

if __name__ == '__main__':
    project_name('RoomFit')
    mem = ['이민섭','박현우','박찬우','우한봄']
    mem.sort()
    members(mem)
    where('Krafton Jungle')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
