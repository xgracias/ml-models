print("Start")

name = input("What is your name? ")
print("Hello " + name)

temperature = 20
print("Today's temperature is " + str(temperature))

if temperature > 18:
    print("It's hot today.")
elif temperature < 5:
    print("It's cold today.")
else:
    print("I don't care about weather.")

print("I'll count from 1 to 5.")
index = 1
while index<=5:
    print(index)
    print(index * "*")
    index += 1

names = ["John", "Bob", "Mosh", "Sam", "Mary"]
print(names)
print(names[0:3])
print(names[-2])

numbers = [1, 2, 3, 4, 5]
numbers.insert(1, 10)
print(numbers)
print(10 in numbers)
print(len(numbers))

for item in numbers:
    print(item)

range_numbers = range(5, 10, 2)
print(range_numbers)
for item in range_numbers:
    print(item)

tuple_nums = (1,2,3)
print(tuple_nums)

print("Done")