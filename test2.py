def test(a,b,c):
	print(a,b,c)
	
a = {"a":1}
b = {"b":2,"c":3}
test(**a,**b)