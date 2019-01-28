import argparse
from notebook.auth import passwd

parser = argparse.ArgumentParser(description='Generate a Jupyter Notebook Password')
parser.add_argument('-p', action="store", dest="password", type=str)

def main():
    args = parser.parse_args()
    print(passwd(args.password))

if __name__ == "__main__":
    main()
