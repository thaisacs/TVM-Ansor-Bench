
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
    filename = 'r18-original.in'
    filename = 'original-10.in'
    filename = 'o.in'
    x = get_ms(filename)
    filename = 'cache-10.in'
    filename = 'c.in'
    y = get_ms(filename)

    print("original:", x)
    print("   cache:", y)
    print(" speedup:", x/y)


