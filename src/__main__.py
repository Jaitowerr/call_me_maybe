import sys



if __name__ == "__main__":
    print("Hello from call-me-maybe!")

    variable = sys.argv
    if variable:
        for var in variable:
            print(var)
