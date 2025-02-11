class RandomClass:
    random_variable = "var"

    def name(self):
        print("Hello")

def main():
    c = RandomClass()
    c.name()

main()



# class Animal:
#     def __init__(self, name, species):
#         self._name = name  # Protected attribute
#         self._species = species  # Protected attribute
#
#     def _get_name(self):
#         return self._name  # Protected method
#
#     def __set_name(self, name):
#         self._name = name  # Protected method to modify the protected attribute
#
#     def speak(self):
#         print(f"{self._name} says hello!")
#
# # Subclass inherits from Animal
# class Dog(Animal):
#     def __init__(self, name, breed):
#         super().__init__(name, species="Dog")
#         self._breed = breed  # Protected attribute
#
#     def speak(self):
#         print(f"{self._name} the {self._breed} says woof!")
#
# # Usage:
# dog = Dog("Max", "Golden Retriever")
# print(dog._name)
# dog._Animal__set_name("Min")
# print(dog._name)


# class Computer:
#     def __init__(self):
#         self.__wholesalePrice = 900
#         self.retailPrice = 1500
#         self._discount = 600
#
# c = Computer()
# print(c.__dict__)
# print(c._Computer__wholesalePrice)
# print(c._discount)
# print(c.retailPrice)


# class person:
#
#     def __init__(self, name, age, gender):
#         self.name = name
#         self.age = age
#         self.gender = gender
#
#
#
# class student(person):
#
#     def __init__(self, name, age, gender, grade, homeroom):
#         super().__init__(name, age, gender)
#         self.grade = grade
#         self.homeroome = homeroom
#
#
# forrest = person(name='Forrest', age=10, gender='Male')
#
# forrest = student(grade=4, homeroom='4D', name='Forrest', age=10, gender='Male')
#
# j = 1



# while True:
#     number = input()
#     print(number)
#     print('Length of input is ' + str(len(number)))


# import multiprocessing
#
# def square_calculation(x):
#     print(x * x)
#     return x*x
#
# procs = []
# for i in range(4):
#     proc_i = multiprocessing.Process(target=square_calculation, args=(i, ))
#     procs.append(proc_i)
#     procs[-1].start()
#
# for i in range(len(procs)):
#     procs[i].join()
#
#
#
# j = 10
#
#
#
# https://chatgpt.com/share/66fb4aa5-1574-800d-be69-b7be2519c712