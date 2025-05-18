def read():
    print()
    x = input ("Introdu numarul dorit de teste: ") 
    while int(x) < 700:
        x = input ("Numar invalid de teste, alege un alt numar: ") 
    return x
