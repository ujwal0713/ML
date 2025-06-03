import csv

a = []
with open('enjoysport.csv', 'r') as csvfile:
    for row in csv.reader(csvfile):
        a.append(row)

print("\nThe total number of training instances are", len(a))

attribute = len(a[0]) - 1

print("\nThe initial Hypothesis is:")
hypothesis = ['0'] * attribute
print(hypothesis)

for i in range(len(a)):
    if a[i][attribute].lower() == 'yes':
        for j in range(attribute):
            if hypothesis[j] == '0':
                hypothesis[j] = a[i][j]
            elif hypothesis[j] != a[i][j]:
                hypothesis[j] = '?'
    print("\nThe hypothesis after instance", i + 1, "is:", hypothesis)

print("\nThe Maximally Specific Hypothesis is:")
print(hypothesis)