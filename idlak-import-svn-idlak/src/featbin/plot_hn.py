import pylab

def load_data(f, key):
    return [map(float, k.strip().split()[1:]) for k in open(f).readlines() if k[0] == key]

HH, NN, CC, OO, FF, BB = map(lambda x: load_data('/tmp/log', x), ['H','N','C', 'O','F', 'B'])

#HH = [k.strip().split()[1:] for k in open('/tmp/log').readlines() if k[0] == 'H']
#NN = [k.strip().split()[1:] for k in open('/tmp/log').readlines() if k[0] == 'N']
#CC = [k.strip().split()[1:] for k in open('/tmp/log').readlines() if k[0] == 'C']
#OO = [filter(float, k.strip().split()[1:]) for k in open('/tmp/log').readlines() if k[0] == 'O']
#FF = [k.strip().split()[1:] for k in open('/tmp/log').readlines() if k[0] == 'F']

i = 0
for o,f,h,n,c,b in zip(OO,FF,HH,NN,CC,BB):
    i += 1
    #if i <= 900:
    #    continue
    pylab.plot(o, 'b')
    pylab.plot(f, 'r')
    pylab.plot(b, 'g')
    pylab.plot(h, 'g')
    #pylab.plot(n, 'y')
    pylab.plot([k * 4 - 4.5 for k in c], 'g')
    pylab.show()

for h,n,c in zip(HH,NN,CC):
    pylab.plot(h, 'b')
    pylab.plot(n, 'r')
    pylab.plot(c, 'g')
    pylab.show()
