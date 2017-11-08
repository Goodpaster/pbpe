all:
	python setup.py build_ext --inplace
clean:
	rm -rfv build *.pyc src/*.so src/*.c src/*.pyc
