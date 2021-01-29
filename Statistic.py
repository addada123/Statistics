class Statistic():

    #Own rank func for spearman rank correlation
    def rank(self, x):
        rank = dict((x, i + 1) for i, x in enumerate(sorted(set(x))))
        return [rank[x] for x in x]

    #Own Error Function
    def erf(self, num):
        pi = 3.141592
        a = 2 / self.sqrt(pi)
        b = a * (num - (num ** 3) / 3 + (num ** 5) / 10 - (num ** 7) / 42 + (num ** 9) / 216) #Taylor series of Error function
        return b

    #Own square root func
    def sqrt(self, num):
        return num ** (1/2)

    #Own factorial func
    def factorial(self, num):
        if num == 1 or num == 0:
            return 1
        return num * self.factorial(num-1)

    #Mean func
    def mean(self, arr):
        return sum(arr) / len(arr)

    #Median func
    def median(self, arr):
        arr.sort()
        ln = len(arr)
        index = (ln - 1) // 2
        if (ln % 2):
            return arr[index]
        else:
            return (arr[index] + arr[index + 1]) / 2.0

    #Mod func
    def mod(self, arr):
        mode = 0
        count, max = 0, 0
        arr.sort()
        current = 0
        for i in arr:
            if (i == current):
                count += 1
            else:
                count = 1
                current = i
            if (count > max):
                max = count
                mode = i
        return mode

    #Weighted Mean given arr X, of N integers with an array W representing the respective weights of X's elements
    def weighted_mean(self, arr, arr2):
        total = 0
        for i in range(len(arr)):
            total += arr[i] * arr2[i]
        return total / sum(arr2)

    #Quartiles
    def quartiles(self, arr, key = None):
        arr.sort()
        Q2 = self.median(arr)
        if len(arr) % 2 == 0:
            Q1 = [i for i in arr if i < Q2]
            Q3 = [i for i in arr if i > Q2]
            Q1 = self.median(Q1)
            Q3 = self.median(Q3)
        else:
            arr.remove(Q2)
            Q1 = [i for i in arr if i < Q2]
            Q3 = [i for i in arr if i > Q2]
            Q1 = self.median(Q1)
            Q3 = self.median(Q3)
        if key == "Upper":
            return Q3
        elif key == "Lower":
            return Q1
        elif key == "Middle":
            return Q2

    #Interquartile Range
    def interquartile_range(self, arr):
        return self.quartiles(arr, key = "Upper") - self.quartiles(arr, key = "Lower")

    #Standart Deviation
    def std_dev(self, arr):
        total = 0
        mean = self.mean(arr)
        for i in arr:
            total += (i - mean) ** 2
        return self.sqrt(total / (len(arr)))

    #Binomial Distribution
    def binomial(self, x, n, p):
        fact = self.factorial(n) / (self.factorial(x) * self.factorial(n - x))
        powers = (p ** x) * ((1 - p) ** (n - x))
        return fact * powers

    #Binomial Distribution for range(a, b)
    def binomial_n(self, n, p, a, b, key = None):
        total = 0
        if key == "Positive":
            for i in range(a, b + 1):
                total += self.binomial(i, n, p)
            return total
        elif key == "Negative":
            for i in range(a, b + 1):
                total += self.neg_binomial(i, n, p)
            return total
        else:
            print("Use a valid key")

    #Negative Binomial Distribution
    def neg_binomial(self, x, n, p):
        fact = self.factorial(n - 1) / (self.factorial(x - 1) * self.factorial(n - x - 2))
        powers = (p ** x) * ((1 - p) ** (n - x))
        return fact * powers

    #Geometric Distribution number of successes is 1
    def geo_dist(self, n, p):
        return  ((p - 1) ** (n - 1)) * p

    #Geometric Distribution for n time
    def geo_dist_n(self, n, p):
        total = 0
        for i in range(1, n + 1):
            total += self.geo_dist(i, p)
        return total

    #Poisson function
    def poisson(self, k, a):
        e = 2.71828
        poisson = ((a ** k) * (e ** -a)) / self.factorial(k)
        return poisson

    #Cumulative probability.
    def cumulative(self, u, a, x):
        cumulative = 1/2 * (1 + self.erf((x - u) / (a * self.sqrt(2))))
        return cumulative


    #Normal Distribution
    def normal(self, u, var, a, x):
        pi = 3.141592
        e = 2.71828
        a = 1 / (a * self.sqrt(2 * pi))
        b = e ** ((-1) * ((x-u) ** 2) / (2 * var))
        return a * b

    #Phi for normal
    def phi_norm(self, x):
        pi = 3.141592
        e = 2.71828
        phi1 = e ** ((-1) * ((x) ** 2) / 2)
        phi2 = self.sqrt(2 * pi)
        phi = phi1 * phi2
        return phi

    #Standard normal if mean = 0 and deviation = 1
    def standard_normal(self, x, u, a):
        normal = (1 / self.sqrt(a)) * self.phi_norm((x - u) / self.sqrt(a))
        return normal

    #Central Limit Theorem
    def central(self, max, unit, u, a):
        new_u = u * unit
        new_a = self.sqrt(unit) * a
        return self.cumulative(new_u, new_a, max)

    #Pearson Correlation Coefficient
    def pearson_correlation_coefficient(self, arr, arr2):
        if len(arr) != len(arr2):
            return "Len of arrays are not equal"
        ux = self.mean(arr)
        length = len(arr)
        uy = self.mean(arr2)
        ax = 0
        ay = 0
        total = 0
        for i in range(length):
            ax += (arr[i] - ux) ** 2
            ay += (arr2[i] - uy) ** 2
        ax = self.sqrt(ax / length)
        ay = self.sqrt(ay / length)
        for i in range(length):
            total += (arr[i] - ux) * (arr2[i] - uy)
        answer = total / (length * ax * ay)
        return answer

    #Spearman's Rank Correlation
    def spearmans_rank(self, rx, ry):
        if len(rx) != len(ry):
            return "Their lengths are not equal"
        rx = self.rank(rx)
        ry = self.rank(ry)
        n = len(rx)
        total = 0
        for i in range(len(rx)):
            total += (rx[i] - ry[i]) ** 2
        answer = 1 - ((6 * total) / (n * (n ** 2 - 1)))
        return answer

    #Linear Regression
    def linear_regression(self, arr, arr2, predict):
        if len(arr) != len(arr2):
            return "Len of arrays are not equal"
        ux = self.mean(arr)
        length = len(arr)
        uy = self.mean(arr2)
        ax = 0
        ay = 0
        for i in range(length):
            ax += (arr[i] - ux) ** 2
            ay += (arr2[i] - uy) ** 2
        ax = self.sqrt(ax / length)
        ay = self.sqrt(ay / length)
        b = self.pearson_correlation_coefficient(arr, arr2) * (ay / ax)
        a = uy - b * ux
        new_y = a + b * predict
        return new_y

    #Multiple Linear Regression
    def multiple_linear_regression(self, arr, arr2):
        pass