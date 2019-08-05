import random
def bit_turnoff(x):
	o1=x-1
	res=x&o1
	return res


x=random.randint(1,10000)
print(bin(x))
res=bit_turnoff(x)

print(bin(res))