pylint:
    find . -path "*.py" | xargs pylint -E;
test:
    py.test
clean:
    - find . -path "*__pycache__" | xargs rm -rf
    - find . -path "*.pyc" | xargs rm -rf
