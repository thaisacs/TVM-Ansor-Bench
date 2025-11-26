
def get_speed(filename):
    with open(filename) as f:
        lines = f.readlines()
    sum_ = 0
    for line in lines:
        x = line.split('|')
        if(len(x) > 4):
            y = x[4].replace(' ', '')
            if(y != '-'):
                sum_ += float(y)
    return sum_

def get_ms(filename):
    with open(filename) as f:
        lines = f.readlines()
    sum_ = 0
    for line in lines:
        x = line.split('|')
        if(len(x) > 4):
            y = x[3].replace(' ', '')
            if(y != '-'):
                sum_ += float(y)
    return sum_

if __name__ == "__main__":
    filename = 'original.in'
    x = get_ms(filename)
    filename = 'cache.in'
    y = get_ms(filename)

    print("original:", x)
    print("   cache:", y)
    print(" speedup:", x/y)


    filename = 'original.in'
    x = get_speed(filename)
    filename = 'cache.in'
    y = get_speed(filename)

    print("original:", x)
    print("   cache:", y)
    print(" speedup:", x/y)


