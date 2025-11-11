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

