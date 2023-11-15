import math
import numpy as np

class Interval:
    def __init__(self, a, b=None):

        if b is None:
            self.b_min = a
            self.b_max = a
        else:
            self.b_min = min(a, b)
            self.b_max = max(a, b)

    def __add__(self, other):
        return Interval(self.b_min + other.b_min, self.b_max + other.b_max)

    def __sub__(self, other):
        return Interval(self.b_min - other.b_max, self.b_max - other.b_min)

    def __mul__(self, other):
        if isinstance(other,Interval):
            p1 = self.b_min * other.b_min
            p2 = self.b_min * other.b_max
            p3 = self.b_max * other.b_min
            p4 = self.b_max * other.b_max
            return Interval(min(p1, p2, p3, p4), max(p1, p2, p3, p4))
        elif isinstance(other,float):
            if other >= 0 :
                return Interval(self.b_min * other,self.b_max * other)
            else:
                return Interval(self.b_max * other,self.b_min * other)

    def scalar_multiply(self, other): 
        if other >= 0 :
            return Interval(self.b_min * other,self.b_max * other)
        else:
            return Interval(self.b_max * other,self.b_min * other)

    def __truediv__(self, other):
        if other.b_min * other.b_max < 0:  # Signs are different -> 0 is in the interval
            if self.b_min != 0 or self.b_max != 0:
                return Interval(float('-inf'), float('inf'))
            else:
                return Interval(math.nan, math.nan)
        elif other.b_min * other.b_max > 0:  # Same signs -> 0 is excluded from the interval
            p1 = self.b_min / other.b_min
            p2 = self.b_min / other.b_max
            p3 = self.b_max / other.b_min
            p4 = self.b_max / other.b_max
            return Interval(min(p1, p2, p3, p4), max(p1, p2, p3, p4))
        elif other.b_min == 0 and other.b_max > 0:
            p1 = math.copysign(float('inf'), self.b_min)
            p2 = self.b_min / other.b_max
            p3 = math.copysign(float('inf'), self.b_max)
            p4 = self.b_max / other.b_max
            return Interval(min(p1, p2, p3, p4), max(p1, p2, p3, p4))
        elif other.b_max == 0 and other.b_min < 0:
            p1 = math.copysign(float('-inf'), self.b_min)
            p2 = self.b_min / other.b_min
            p3 = math.copysign(float('-inf'), self.b_max)
            p4 = self.b_max / other.b_min
            return Interval(min(p1, p2, p3, p4), max(p1, p2, p3, p4))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return Interval(other) / self

    def sqrt(self):
        if self.b_min > 0:
            return Interval(math.sqrt(self.b_min), math.sqrt(self.b_max))
        elif self.b_max > 0:
            return Interval(0, math.sqrt(self.b_max))
        else:
            return EMPTY_SET

    def __str__(self):
        return f"[{self.b_min}, {self.b_max}]"

    def inflate(self, number):
        return Interval(self.b_min - number, self.b_max + number)

class Interval2Vector: 
    def __init__(self, x, y): 
        self.x = Interval(x[0], x[1]) 
        self.y = Interval(y[0], y[1])
    
    def __add__(self, other): 
        return Interval2Vector(self.x + other.x, self.y + other.y)

    def __mul__(self, other):
        if isinstance(other, Interval2Vector):
            return Interval2Vector(self.x * other.x, self.y * other.y)
        elif isinstance(other, float):
            return Interval2Vector(self.x * other, self.y * other)
        else:
            raise ValueError("Unsupported multiplication operation")

    def inflate(self,eps):
        return Interval2Vector(self.x.inflate(eps),self.y.inflate(eps))
    
    def mid(self): 
        return np.array([[self.x.b_min + self.x.b_max], [self.y.b_min + self.y.b_max]]) * 0.5 
    
    def uncertainty(self): 
        l1 = np.abs(self.x.b_max - self.x.b_min)
        l2 = np.abs(self.y.b_max - self.y.b_min)
        return np.max(np.array([l1, l2]))

oo = float('inf')
EMPTY_SET = Interval(float('nan'), float('nan'))

# Code that runs when the script is executed directly
if __name__ == "__main__":
    # Example usage of the Interval class
    interval1 = Interval(2, 5)
    interval2 = Interval(1, 3)

    # Perform some interval operations
    sum_result = interval1 + interval2
    difference_result = interval1 - interval2
    product_result = interval1 * interval2
    division_result = interval1 / interval2

    # Display the results
    print("Interval1:", interval1.b_min, interval1.b_max)
    print("Interval2:", interval2.b_min, interval2.b_max)
    print("Sum:", sum_result.b_min, sum_result.b_max)
    print("Difference:", difference_result.b_min, difference_result.b_max)
    print("Product:", product_result.b_min, product_result.b_max)
    print("Division:", division_result.b_min, division_result.b_max)
