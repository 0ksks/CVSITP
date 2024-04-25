with open("UECFOODPIXCOMPLETE/category.txt","r") as f:
    a = f.read()
a = a.split("\n")
a = list(map(lambda x:x.split("\t"), a))
a = a[1:]
for aa in a:
    print(f"\t{int(aa[0])-1}: {aa[1]}")