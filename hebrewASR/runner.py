import sys
from streamlit.web import cli as stcli


def main():
    sys.argv = ["streamlit", "run", "hebrewASR/app.py"]
    sys.exit(stcli.main())


if __name__ == '__main__':
    main()