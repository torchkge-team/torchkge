# -*- coding: utf-8 -*-
"""
Copyright Armand Boschin
armand.boschin@telecom-paristech.fr
"""


class MyClass2:
    """A whatever-you-are-doing.

    :param a: the `a` of the system.
    :param b: the `b` of the system.

    >>> my_object = MyClass2(a = 5, b = 3)
    """

    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def addition(self) -> float:
        """
        Add :attr:`a` and :attr:`b`.

        :return: :attr:`a` + :attr:`b`.

        >>> my_object = MyClass2(a=5, b=3)
        >>> my_object.addition()
        8
        """
        return self.a + self.b


if __name__ == '__main__':
    print('Do some little tests here')
    test = MyClass2(a=42, b=51)
    print(test.addition())
