all:
	python setup.py build_ext --inplace
clean:
	rm -rf *.so *.c build
