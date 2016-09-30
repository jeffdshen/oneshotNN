import random

def main():
  seed = 34958
  with open('data/data.txt', 'w') as file:
    x = 1000000000000000
    for i in range(0, 1000):
      a = random.randint(0, x)
      b = random.randint(0, x)
      file.write(str(a))
      file.write('+')
      file.write(str(b))
      file.write('=')
      file.write(str(a + b))
      file.write('\n')

if __name__ == '__main__':
  main()