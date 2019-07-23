# train:
#	python work/train_mnist.py

doc:
	sphinx-apidoc -f -o docs/ src/
	sphinx-build -b html ./docs ./docs/_build

test:
	pytest -v --capture=no

lint:
	flake8 ./src/

format:
	autopep8 --in-place --aggressive --aggressive --recursive ./src/
