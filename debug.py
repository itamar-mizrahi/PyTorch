import logging

# Python program with errors fixed

logging.basicConfig(filename="app.log",  level=logging.DEBUG)

var = 8
sumvalue = var + 4
def dosomething(valuetocheck):
    if valuetocheck > 4:
        logging.debug(f"Value {valuetocheck} is greater than 4.")
        print("Indent fixed")

dosomething(sumvalue)

def test_dosomething():
    # Test case 1: valuetocheck is greater than 4
    dosomething(5)
    # Add assertions here if you want to check specific outcomes,
    # e.g., checking log output or return values.

    # Test case 2: valuetocheck is not greater than 4
    dosomething(3)

if __name__ == '__main__':
    test_dosomething()

#test