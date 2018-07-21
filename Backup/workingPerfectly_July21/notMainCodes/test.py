import sys

gpuNum = 1
IxNuclei = 1
for input in sys.argv:
    if input.split('=')[0] == 'ind':
        IxNuclei = int(input.split('=')[1])
    elif input.split('=')[0] == 'gpu':
        gpuNum = int(input.split('=')[1])

print('gpuNumg',gpuNum)
print('IxNuclei',IxNuclei)
